"""Run handcrafted Native on a HeXO Bot API site from this computer."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import getpass
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import aiohttp

from hexo import Game, library


def answer(history, packet, ms, width, depth):
    """Apply confirmed turns only; our suggestion arrives in a later previous."""
    game = Game(history)
    try:
        for turn in packet['previous']:
            if turn['side'] != ('x', 'o')[game.player]:
                raise ValueError('Confirmed turn has the wrong side')
            for cell in turn['pieces']:
                game.play(cell['q'], cell['r'])
        if packet['side'] != ('x', 'o')[game.player]:
            raise ValueError('Request has the wrong side')
        confirmed = [c[:2] for c in game.cells]
        if 'move_time_limit' in packet:
            ms = min(ms, max(1, int(packet['move_time_limit'] * 1000) - 150))
        result = game.search(ms=ms, width=width, depth=depth)
        for move in result['moves']:
            game.play(*move)
        response = {'type': 'move_response', 'request_id': packet['request_id'],
                    'move': {'pieces': [{'q': q, 'r': r} for q, r in result['moves']]}}
        return confirmed, response, result
    finally:
        game.close()


class Refused(Exception):
    def __init__(self, status, retry=1):
        self.status, self.retry = status, retry
        super().__init__(f'HTTP {status}')


def retry_delay(headers):
    return max(1, int(headers.get('Retry-After', '1')))


class NativeArena:
    def __init__(self, url, token, *, ms=100, width=16, depth=12):
        self.url = url.rstrip('/')
        self.token = token
        self.ms, self.width, self.depth = ms, width, depth
        self.games = {}
        self.challenges = set()
        self.presence = None

    async def call(self, method, path, body=None):
        for attempt in range(3):
            async with self.http.request(method, self.url + path, json=body,
                                         headers={'Authorization': 'Bearer ' + self.token},
                                         timeout=aiohttp.ClientTimeout(total=30),
                                         allow_redirects=False) as response:
                if response.status < 300:
                    return await response.json()
                delay = retry_delay(response.headers)
                if response.status not in (429, 503) or attempt == 2:
                    raise Refused(response.status, delay)
            await asyncio.sleep(delay)

    async def accept(self, challenge):
        challenge_id = challenge['challengeId']
        try:
            await self.call('POST', f'/api/bot/challenge/{quote(challenge_id, safe="")}/accept')
            print(f'Accepted challenge from {challenge["challenger"]["name"]}', flush=True)
        except (Refused, aiohttp.ClientError, asyncio.TimeoutError) as error:
            print(f'Challenge not accepted: {type(error).__name__}', flush=True)

    async def session(self, socket, worker, game_id):
        history = []
        search = None
        incoming = asyncio.create_task(socket.receive())

        async def respond(packet):
            computation = asyncio.get_running_loop().run_in_executor(
                worker, answer, history, packet, self.ms, self.width, self.depth)
            try:
                confirmed, reply, result = await asyncio.shield(computation)
            finally:
                # Cancelling an asyncio wait cannot stop Native. Finish its bounded
                # work before redialling, then receive the server's updated clock.
                await asyncio.gather(computation, return_exceptions=True)
            await socket.send_json(reply)
            print(f'Game {game_id}: ply {len(confirmed)}, '
                  f'{result["elapsed_ms"]:.1f} ms, depth {result["depth"]}', flush=True)
            return confirmed

        try:
            while True:
                waiting = [incoming] + ([search] if search is not None else [])
                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                busy = search is not None
                if search in done:
                    history = search.result()
                    search = None
                if incoming not in done:
                    continue
                message = incoming.result()
                if message.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError('Engine socket failed')
                if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                                    aiohttp.WSMsgType.CLOSED):
                    return
                if message.type == aiohttp.WSMsgType.TEXT:
                    packet = json.loads(message.data)
                    if packet['type'] == 'setup':
                        if packet['board']['cells'] != [{'q': 0, 'r': 0, 'p': 'x'}]:
                            raise ValueError('Expected the Arena origin setup')
                        history = [[0, 0]]
                    elif packet['type'] == 'move_request':
                        if search is not None:
                            raise ValueError('Overlapping move requests')
                        search = asyncio.create_task(respond(packet))
                    elif packet['type'] == 'heartbeat' and packet['waiting'] and not busy:
                        raise ConnectionError('Server is waiting on an idle session')
                incoming = asyncio.create_task(socket.receive())
        finally:
            pending = [incoming] + ([search] if search is not None else [])
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def play(self, event):
        game_id = event['gameId']
        endpoint = urlsplit(urljoin(self.url + '/', event['engine']['socketUrl']))
        origin = urlsplit(self.url)
        if (endpoint.scheme, endpoint.netloc) != (origin.scheme, origin.netloc):
            raise ValueError('Engine socket must be on the API origin')
        socket_url = urlunsplit(('wss' if endpoint.scheme == 'https' else 'ws',
                                endpoint.netloc, endpoint.path, '', ''))
        # A dedicated thread retains Native's proven continuation for this game.
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='native-game')
        backoff = 1
        try:
            while True:
                delay = backoff
                try:
                    async with self.http.ws_connect(socket_url,
                            params={'token': event['engine']['token']}, max_msg_size=65536) as socket:
                        backoff = 1
                        await self.session(socket, worker, game_id)
                        if socket.close_code in (1000, 1001):
                            return
                except aiohttp.WSServerHandshakeError as error:
                    if error.status == 404:
                        # Reopen presence to replay live games with fresh tokens.
                        if self.presence is not None:
                            self.presence.close()
                        return
                    if error.status not in (429, 503):
                        raise Refused(error.status) from None
                    delay = max(delay, retry_delay(error.headers))
                except (aiohttp.ClientError, ConnectionError, asyncio.TimeoutError):
                    print(f'Game {game_id}: reconnecting', flush=True)
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, 8)
        finally:
            worker.shutdown(wait=False, cancel_futures=True)

    async def event(self, event):
        kind = event['type']
        if kind == 'gameStart':
            game_id = event['gameId']
            active = self.games.get(game_id)
            if active is not None and not active.done():
                return
            await self.finish(game_id)
            print(f'Game {game_id} vs {event["opponent"]["name"]}, '
                  f'playing {event["side"]}', flush=True)
            self.games[game_id] = asyncio.create_task(self.play(event))
            self.games[game_id].add_done_callback(lambda task: self.game_done(game_id, task))
        elif kind == 'gameFinish':
            print(f'Game {event["gameId"]} finished: {event["reason"]}, '
                  f'winner {event["winner"]}', flush=True)
            await self.finish(event['gameId'])
        elif kind == 'challenge':
            task = asyncio.create_task(self.accept(event['challenge']))
            self.challenges.add(task)
            task.add_done_callback(self.challenges.discard)
        # Presence moveRequest is informational; the socket supplies the request.

    @staticmethod
    def game_done(game_id, task):
        if not task.cancelled() and (error := task.exception()) is not None:
            import traceback
            frame = traceback.extract_tb(error.__traceback__)[-1]
            print(f'Game {game_id} stopped: {type(error).__name__} '
                  f'at {Path(frame.filename).name}:{frame.lineno}', flush=True)

    async def finish(self, game_id):
        task = self.games.pop(game_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15)
        async with aiohttp.ClientSession(timeout=timeout) as self.http:
            account = await self.call('GET', '/api/bot/account')
            digest = hashlib.sha256(Path(library).read_bytes()).hexdigest()
            await self.call('PATCH', '/api/bot/account', {
                'about': f'Handcrafted Native, CPU search, {self.ms} ms per turn. No neural model.',
                'version': digest[:12], 'repoUrl': 'https://github.com/Tomodovodoo/HeXO',
                'accepts': {'turnMs': [5000, 600000], 'match': True, 'unlimited': True}})
            print(f'{account["name"]}: {self.url}/bots/{quote(account["name"])}', flush=True)
            print(f'Native library {digest}, {self.ms} ms per turn', flush=True)
            backoff = 1
            try:
                while True:
                    delay = backoff
                    try:
                        async with self.http.get(self.url + '/api/bot/stream?open=1',
                                headers={'Authorization': 'Bearer ' + self.token},
                                allow_redirects=False) as response:
                            self.presence = response
                            if response.status != 200:
                                raise Refused(response.status, retry_delay(response.headers))
                            print('Online and accepting games', flush=True)
                            backoff = 1
                            async for line in response.content:
                                if line.strip():
                                    await self.event(json.loads(line))
                    except Refused as error:
                        if error.status not in (429, 503):
                            raise
                        delay = max(delay, error.retry)
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        print('Presence stream disconnected', flush=True)
                    await asyncio.sleep(delay)
                    backoff = min(backoff * 2, 8)
            finally:
                for task in self.challenges:
                    task.cancel()
                await asyncio.gather(*self.challenges, return_exceptions=True)
                for game_id in list(self.games):
                    await self.finish(game_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url', nargs='?', default='https://hexo.seeligto.de')
    parser.add_argument('--ms', type=int, default=100)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--depth', type=int, default=12)
    args = parser.parse_args()
    origin = urlsplit(args.url)
    if (origin.scheme not in ('https', 'http') or not origin.netloc or origin.path not in ('', '/')
            or origin.username or origin.password or origin.query or origin.fragment):
        parser.error('Supply an HTTP(S) site origin')
    if not 1 <= args.ms <= 30000 or not 2 <= args.width <= 128 or not 1 <= args.depth <= 64:
        parser.error('Invalid Native search budget')
    token = os.environ.get('HEXO_TOKEN') or getpass.getpass('Bot token: ')
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    try:
        asyncio.run(NativeArena(args.url, token, ms=args.ms, width=args.width, depth=args.depth).run())
    except KeyboardInterrupt:
        pass
    except (Refused, aiohttp.ClientError) as error:
        # Transport exception strings can contain the private game-token URL.
        reason = f'HTTP {error.status}' if isinstance(error, Refused) else type(error).__name__
        raise SystemExit(f'Arena connection failed: {reason}') from None


if __name__ == '__main__':
    main()
