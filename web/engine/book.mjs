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
 * Wraps `page.post`. A /book request through it that succeeds marks the switch touched for the session (`storage`,
 * sessionStorage in the page); a /seat request that names an engine is followed, while untouched, by the /book request
 * that applies `bookDefault`. `state()` is the page's current state, `human(seat)` whether the page shows that seat
 * object as Human.
 */
export function followSeats(page, state, human, storage) {
  const post = page.post;
  let touched = false;
  try { touched = storage?.getItem(KEY) === '1'; } catch {}
  page.post = async (path, body = {}) => {
    const data = await post(path, body);
    if (data && path === '/book') {
      touched = true;
      try { storage?.setItem(KEY, '1'); } catch {}
    }
    const s = state();
    if (data && path === '/seat' && body.engine !== undefined && !touched && s?.book?.available) {
      const enabled = bookDefault(s.seats.map(human));
      if (enabled !== s.book.enabled) await post('/book', {enabled});
    }
    return data;
  };
}
