"""Bubble's HTTTX bot socket, clocked match API and live clock stream."""
import argparse
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from aiohttp import web
from bot_api import board_game, APIError, _coord
from hexo import Game
from timed_engine import TimedEngine, legal_turn
from timed_match import Match, play_turn
from time_control import milliseconds


def create_app(default_config=None, *, run=None, directory=None, engine_factory=TimedEngine):
    default_config = dict(default_config or dict(kind='drip'))
    matches, engines, tasks = {}, {}, set()
    sockets = set()
    shutting_down = asyncio.Event()

    @web.middleware
    async def errors(request, handler):
        try:
            return await handler(request)
        except APIError as error:
            return web.json_response(dict(error=str(error)), status=error.status)
        except (ValueError, KeyError, TypeError) as error:
            return web.json_response(dict(error=str(error)), status=400)

    app = web.Application(middlewares=[errors], client_max_size=1_048_576)
    engine_key = web.AppKey('default_engine', TimedEngine)
    watchdog_key = web.AppKey('watchdog', asyncio.Task)
    def spawn(coro):
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    def config(player):
        if player.get('kind') == 'human':
            return dict(player)
        configured = default_config | player
        if run and configured.get('kind') == 'bubble':
            configured['run'] = str(Path(run).resolve())
        return configured

    async def start_engine(configuration):
        task = asyncio.create_task(asyncio.to_thread(engine_factory, configuration))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            engine = await task
            await asyncio.to_thread(engine.close)
            raise

    def identities(match, prepared):
        with match.lock:
            previous = match.specification.get('identities')
            current = [engine.identity if engine else dict(checkpoint='human') for engine in prepared]
            match.specification['identities'] = current
            if match.directory:
                match._write('spec.json', match.specification)
            match.record('engines', identities=current, previous_identities=previous)

    async def prepare(match):
        prepared = []
        try:
            for side in ('cross', 'circle'):
                player = match.specification['players'][side]
                prepared.append(None if player['kind'] == 'human' else
                                await start_engine(player))
            engines[match.id] = prepared
            identities(match, prepared)
            with match.lock:
                match.state = 'ready'
                match.record('ready')
        except BaseException as error:
            engines.pop(match.id, None)
            for engine in prepared:
                if engine:
                    await asyncio.to_thread(engine.close)
            if isinstance(error, asyncio.CancelledError):
                raise
            with match.lock:
                match.state = 'failed'
                match.record('error', error=str(error))

    async def play(match):
        while match.tick()['state'] == 'playing':
            with match.lock:
                engine = engines[match.id][match.game.player]
                cancellation = match.cancel
            if engine is None:
                return
            await asyncio.to_thread(play_turn, match, engine)
            if cancellation.is_set():
                return
            await asyncio.sleep(0)

    async def release(match):
        playing = getattr(match, 'playing_task', None)
        if playing and playing is not asyncio.current_task():
            with suppress(Exception):
                await playing
        for engine in engines.pop(match.id, []):
            if engine:
                await asyncio.to_thread(engine.close)

    def schedule(match):
        if match.state == 'finished':
            releasing = getattr(match, 'release_task', None)
            if match.id in engines and (releasing is None or releasing.done()):
                match.release_task = spawn(release(match))
            return
        existing = getattr(match, 'playing_task', None)
        if existing and not existing.done():
            # An abandoned turn's thread exits before a resumed request starts.
            existing.add_done_callback(lambda _: schedule(match) if match.state == 'playing' else None)
        elif match.state == 'playing':
            match.playing_task = spawn(play(match))

    async def capabilities(request):
        return web.json_response(dict(meta=dict(name='Bubble', version='1',
            **{'x-bubble': dict(clock_config='v1', history_config='v1', match_api='v1')}),
            stateless=dict(versions={'v1-alpha': dict(request_id=True, move_time_limit=True)}),
            basic_websocket=dict(versions={'v1-alpha': dict(move_time_limit=True, request_id=True,
                interruptible=True, evaluation=True, evaluation_time_limit=True,
                config=dict(dynamic=True), resettable_state=True, move_skips=True, dual_sided=True)})))

    async def models(request):
        if not run:
            return web.json_response([])
        from dense_player import DensePlayer
        selected = DensePlayer.models(SimpleNamespace(run=Path(run), model_path=None))
        return web.json_response(selected)

    async def stateless(request):
        started = time.monotonic()
        body = await request.json()
        limit = milliseconds(body.get('time_limit', 1), 'time_limit')*1000
        if limit <= 0:
            raise APIError('No processing time was provided', 408)
        ident = body.get('request_id')
        if ident is not None and (type(ident) is not int or ident < 0):
            raise ValueError('request_id must be nonnegative')
        game = board_game(body['board'], deadline=started+min(1, limit/1000))
        try:
            remaining = max(0, limit-(time.monotonic()-started)*1000)
            result = await asyncio.to_thread(app[engine_key].turn, game, remaining)
            response = dict(move=dict(pieces=[dict(q=q, r=r) for q, r in result['moves']]))
            if ident is not None:
                response['request_id'] = ident
            return web.json_response(response)
        finally:
            game.close()

    async def turn(request):
        started = time.monotonic()
        body = await request.json()
        game = Game(body['history'])
        try:
            clock = body.get('clock')
            cap = body.get('time_limit_ms')
            if cap is not None:
                cap = max(0, milliseconds(cap, 'time_limit_ms')-(time.monotonic()-started)*1000)
            if clock:
                clock = dict(clock)
                field = 'cross_ms' if game.player == 0 else 'circle_ms'
                clock[field] = max(0, milliseconds(clock[field], field)-(time.monotonic()-started)*1000)
            result = await asyncio.to_thread(app[engine_key].turn, game, cap, clock=clock)
            return web.json_response(result)
        finally:
            game.close()

    async def socket(request):
        ws = web.WebSocketResponse(max_msg_size=1_048_576, heartbeat=20)
        await ws.prepare(request)
        sockets.add(ws)
        engine = None
        history, saved, clock, setup_history = [[0, 0]], None, None, None
        generation, last_id, pending = 0, -1, None
        cancellation = threading.Event()

        async def answer(local, ident, token, budget, clock_state, evaluate, cancel_event):
            nonlocal history, saved
            game = Game(local)
            try:
                result = await asyncio.to_thread(engine.turn, game, budget, clock=clock_state,
                                                cancel=cancel_event)
                if token != generation or cancel_event.is_set():
                    return
                if evaluate:
                    value = result.get('win_probability')
                    evaluation = {} if value is None else dict(heuristic=(2*value-1)*(1 if game.player == 0 else -1))
                    packet = dict(type='eval_response', evaluation=evaluation)
                    if ident is not None:
                        packet['request_id'] = ident
                    await ws.send_json(packet)
                else:
                    legal_turn(local, result['moves'])
                    packet = dict(type='move_response', move=dict(pieces=[dict(q=q, r=r) for q, r in result['moves']]))
                    if ident is not None:
                        packet['request_id'] = ident
                    await ws.send_json(packet)
                    if token == generation and not cancel_event.is_set():
                        history, saved = local, None
            except Exception as error:
                if token == generation and not cancel_event.is_set():
                    await ws.send_json(dict(type='error', error=str(error), request_id=ident))
            finally:
                game.close()

        try:
            engine = await start_engine(default_config)
            async for message in ws:
                if message.type != web.WSMsgType.TEXT:
                    continue
                try:
                    packet = json.loads(message.data)
                    kind = packet['type']
                    if kind == 'interrupt':
                        if packet.get('request_id') is not None and packet['request_id'] != last_id:
                            continue
                        generation += 1
                        cancellation.set()
                        if saved is not None:
                            history, saved = saved, None
                    elif kind == 'config':
                        if pending and not pending.done() and not cancellation.is_set():
                            raise ValueError('Configuration requires an idle session')
                        if 'x-bubble-clock' in packet:
                            clock = dict(packet['x-bubble-clock'])
                            if clock.pop('version', 1) != 1:
                                raise ValueError('Unsupported clock version')
                            for field in ('cross_ms', 'circle_ms', 'increment_ms'):
                                milliseconds(clock[field], field)
                            clock['_received'] = time.monotonic()
                        if 'x-bubble-history' in packet:
                            setup_history = packet['x-bubble-history']
                    elif kind == 'setup':
                        generation += 1
                        cancellation.set()
                        if setup_history is not None:
                            game = Game(setup_history)
                        else:
                            board = packet.get('board', dict(cells=[dict(q=0, r=0, p='x')]))
                            if board['cells'] != [dict(q=0, r=0, p='x')]:
                                raise ValueError('Use negotiated x-bubble-history for non-origin setup')
                            game = Game([[0, 0]])
                        try:
                            history = [list(c[:2]) for c in game.cells]
                        finally:
                            game.close()
                        saved, setup_history, clock = None, None, None
                    elif kind in ('move_request', 'eval_request'):
                        if pending and not pending.done() and not cancellation.is_set():
                            raise ValueError('One request at a time')
                        ident = packet.get('request_id')
                        if ident is not None:
                            if type(ident) is not int or ident <= last_id:
                                raise ValueError('request_id must increase within the connection')
                            last_id = ident
                        local = history.copy()
                        if kind == 'move_request':
                            for turn in packet['previous']:
                                probe = Game(local)
                                try:
                                    if turn['side'] != ('x', 'o')[probe.player]:
                                        raise ValueError('Previous move has the wrong side')
                                finally:
                                    probe.close()
                                moves = [_coord(cell) for cell in turn['pieces']]
                                legal_turn(local, [list(p) for p in moves])
                                local.extend(list(p) for p in moves)
                        probe = Game(local)
                        try:
                            if packet['side'] != ('x', 'o')[probe.player]:
                                raise ValueError('Requested side disagrees with the position')
                        finally:
                            probe.close()
                        saved = history.copy()
                        generation += 1
                        cancellation = threading.Event()
                        evaluate = kind == 'eval_request'
                        field = 'evaluation_time_limit' if evaluate else 'move_time_limit'
                        budget = milliseconds(packet[field], field)*1000 if field in packet else None
                        clock_state, clock = clock, None
                        if clock_state:
                            elapsed = (time.monotonic()-clock_state.pop('_received'))*1000
                            side_key = 'cross_ms' if packet['side'] == 'x' else 'circle_ms'
                            clock_state[side_key] = max(0, clock_state[side_key]-elapsed)
                        pending = spawn(answer(local, ident, generation, budget, clock_state, evaluate, cancellation))
                    elif kind == 'heartbeat':
                        if packet.get('waiting') and (pending is None or pending.done()):
                            await ws.close()
                    else:
                        raise ValueError('Unknown packet type')
                except (ValueError, KeyError, TypeError) as error:
                    await ws.send_json(dict(type='error', error=str(error)))
        finally:
            sockets.discard(ws)
            generation += 1
            cancellation.set()
            try:
                if pending:
                    with suppress(Exception):
                        await pending
            finally:
                # Server shutdown can cancel this handler while it waits for its last answer; the engine still closes.
                if engine:
                    await asyncio.to_thread(engine.close)
        return ws

    async def create(request):
        body = await request.json()
        body['players'] = {side: config(body['players'][side]) for side in ('cross', 'circle')}
        match = Match(body, directory=directory)
        match.state = 'preparing'
        matches[match.id] = match
        match.record('preparing')
        spawn(prepare(match))
        return web.json_response(match.snapshot(), status=202)

    def get(request):
        match = matches.get(request.match_info['id'])
        if match is None:
            raise web.HTTPNotFound()
        return match

    async def state(request):
        return web.json_response(get(request).tick())

    async def action(request):
        match = get(request)
        body = await request.json() if request.can_read_body else {}
        command = request.match_info['action']
        if command == 'start':
            match.start()
        elif command == 'pause':
            match.pause()
        elif command == 'resume':
            if match.id not in engines:
                # Restoration is inert until an explicit resume request.
                with match.lock:
                    if match.state != 'paused':
                        raise ValueError('Match is not paused')
                    match.state = 'preparing'
                prepared = []
                try:
                    for side in ('cross', 'circle'):
                        player = match.specification['players'][side]
                        prepared.append(None if player['kind'] == 'human' else
                                        await start_engine(player))
                    identities(match, prepared)
                except BaseException:
                    for engine in prepared:
                        if engine:
                            await asyncio.to_thread(engine.close)
                    match.state = 'paused'
                    raise
                engines[match.id] = prepared
                match.state = 'paused'
            match.resume()
        elif command == 'resign':
            match.resign(('x', 'o').index(body['side']))
        elif command in ('turn', 'place'):
            with match.lock:
                received = match.clock.now()
                if match.specification['players'][('cross', 'circle')[match.game.player]]['kind'] != 'human':
                    raise ValueError('This side is controlled by an engine')
                pieces = body['pieces'] if command == 'turn' else [[body['q'], body['r']]]
                match.submit(pieces, body['turn_id'], body['revision'], partial=command == 'place', received=received)
        else:
            raise web.HTTPNotFound()
        schedule(match)
        return web.json_response(match.tick())

    async def notation(request):
        return web.Response(text=get(request).notation(), content_type='text/plain')

    async def events(request):
        match = get(request)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache'})
        await response.prepare(request)
        try:
            while not shutting_down.is_set():
                event = dict(type='clock', **match.tick())
                await response.write(('data: '+json.dumps(event, allow_nan=False)+'\n\n').encode())
                if match.state == 'finished':
                    break
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(shutting_down.wait(), .25)
        except (ConnectionError, asyncio.CancelledError):
            pass
        return response

    async def watchdog():
        while True:
            for match in list(matches.values()):
                match.tick()
                if match.state == 'finished':
                    schedule(match)
            await asyncio.sleep(.01)

    async def startup(app):
        if directory and Path(directory).exists():
            for saved in Path(directory).glob('*/state.json'):
                match = Match.restore(saved.parent)
                matches[match.id] = match
        app[engine_key] = await start_engine(default_config)
        app[watchdog_key] = asyncio.create_task(watchdog())

    async def shutdown(app):
        shutting_down.set()
        await asyncio.gather(*(ws.close(code=1001, message=b'Server shutdown') for ws in list(sockets)))

    async def cleanup(app):
        app[watchdog_key].cancel()
        with suppress(asyncio.CancelledError):
            await app[watchdog_key]
        for match in matches.values():
            if match.state == 'playing':
                with suppress(ValueError):
                    match.pause()
            match.cancel.set()
        await asyncio.gather(*list(tasks), return_exceptions=True)
        for prepared in engines.values():
            for engine in prepared:
                if engine:
                    await asyncio.to_thread(engine.close)
        await asyncio.to_thread(app[engine_key].close)
        for match in matches.values():
            match.close()

    app.on_startup.append(startup)
    app.on_shutdown.append(shutdown)
    app.on_cleanup.append(cleanup)
    app.add_routes([web.get('/capabilities.json', capabilities), web.get('/models', models),
                    web.post('/bot', turn), web.post('/analyze', turn),
                    web.post('/stateless/v1-alpha/turn', stateless), web.get('/bws/v1-alpha/game', socket),
                    web.post('/matches', create), web.get('/matches/{id}', state),
                    web.get('/matches/{id}/events', events), web.get('/matches/{id}/notation', notation),
                    web.post('/matches/{id}/{action}', action)])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--net-kernels', choices=['fused', 'reference'], default='fused')
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--out', type=Path, default=Path('artifacts/timed-matches'))
    args = parser.parse_args()
    config = dict(kind='drip')
    if args.run or args.model:
        config = dict(kind='bubble', run=str((args.run or Path('.')).resolve()),
                      device=args.device, net_kernels=args.net_kernels)
        if args.model:
            config['model'] = str(args.model.resolve())
        if args.checkpoint:
            config['checkpoint'] = args.checkpoint
    web.run_app(create_app(config, run=args.run, directory=args.out), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
