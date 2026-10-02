/* Opening-book default for web/index.html: until the person changes the opening-book switch or its set in this
 * browser session, each seat change sets the book from the seats as the page shows them (a Bubble (browser) seat is
 * an engine). The switch is applied through the same /book request it sends. */
const KEY = 'book-touched';

/** Whether the book is on by default for seats where `humans[side]` says the page shows a Human: off when both are,
 * on otherwise (between engines, and the server's default for a person against an engine). */
export function bookDefault(humans) {
  return !humans.every(Boolean);
}

/**
 * Wraps `page.post`. A /book request through it marks the switch touched for the session (`storage`, sessionStorage
 * in the page) when it is sent. /seat requests that name an engine run one at a time, each followed, while untouched,
 * by the /book request that applies `bookDefault` to the seats then shown, so the book follows the last seats.
 * `state()` is the page's current state, `human(seat)` whether the page shows that seat object as Human.
 */
export function followSeats(page, state, human, storage) {
  const post = page.post;
  let touched = false, queue = Promise.resolve();
  try { touched = storage?.getItem(KEY) === '1'; } catch {}
  const follow = async body => {
    const data = await post('/seat', body), s = state();
    if (data && !touched && s?.book?.available) {
      const enabled = bookDefault(s.seats.map(human));
      if (enabled !== s.book.enabled) await post('/book', {enabled});
    }
    return data;
  };
  page.post = (path, body = {}) => {
    if (path === '/book') {
      touched = true;
      try { storage?.setItem(KEY, '1'); } catch {}
    }
    if (path !== '/seat' || body.engine === undefined) return post(path, body);
    const next = queue.then(() => follow(body));
    queue = next.catch(() => {});
    return next;
  };
}
