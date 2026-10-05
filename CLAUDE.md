# ardecode.github.io

Personal site and blog (Jekyll 3.10 via the `github-pages` gem, deployed from `main` by GitHub Pages).

- `index.html` is the homepage. It has empty front matter so Jekyll runs Liquid in it (the Latest Posts section). Avoid `{{` / `{%` anywhere else in that file.
- `projects.html` is the projects page. `blog.html` lists all posts at `/blog/`.
- `_posts/` holds published posts. `_drafts/` holds work in progress (not built on GitHub Pages, but the repo is public, so drafts are visible on GitHub).
- Permalinks are `/:categories/:year/:month/:day/:title/`. Changing a post's categories or filename changes its URL, so don't do that to published posts.

## Writing a post

1. Start in `_drafts/<slug>.md` (no date in the filename).
2. Preview: `bundle exec jekyll serve --drafts --livereload`, then open http://localhost:4000.
3. To publish, move it to `_posts/YYYY-MM-DD-<slug>.md` and push to `main`.

Front matter:

```yaml
---
layout: post
title: "Short, Specific Title"
date: YYYY-MM-DD
categories: [Topic, Subtopic]
description: "One sentence for search results and link previews."
---
```

The first paragraph is the excerpt shown on the homepage and `/blog/`, so make it the hook.

## Voice and style

All writing on this site (posts, About Me, project and travel blurbs) should sound like Arijit, not like generic AI copy. The reference is his older blog, https://ajsviewpoint.blogspot.com/: keep the voice, but not its typos or grammar slips.

What his writing does:

- Opens with a short, punchy line or fragment, often a number. "Thirty minutes. That's the time it took..." / "84 ms to 167 ms."
- First person. Says why he looked into something ("I kept reading about X without having it in my head, so I wrote it down").
- Asks the question a reader would ask, then answers it bluntly. "Do all 10 tokens go into one FFN? No." / "Overconfident? Not at all."
- Starts sentences with "And", "But" and "So".
- Works through the numbers in the open, often introduced with "Think about this."
- Uses numbered lists in a "Label: explanation" form.
- Takes a position and says "I think" or "my read" when it's opinion.
- Ends with a short one-line close, not a summary paragraph.
- Plain words. Short sentences. Indian English is fine.

Never:

- Emoji in headings or body text.
- Filler like "In this blog, we'll take you on a journey", "Thanks for reading!", "Final Thoughts".
- Hype words: seamless, cutting-edge, robust, comprehensive, intelligent, leverage, unlock, delve, vibrant, serene, pristine, mesmerizing.
- Em dashes. Use a full stop, a comma or parentheses.
- Bolding every other phrase. Bold at most a key number or two.
- Made-up personal experiences or opinions. If an anecdote or stance would help, ask him for it or mark it clearly for him to confirm.

## Technical posts

- Use concrete numbers and real configs (model sizes, hardware, commands, before/after tables).
- Check the math and the facts. Flag anything you can't verify instead of guessing.
- Separate what was measured from what is interpretation ("my read").
- Strip internal hostnames, internal model names and anything employer-confidential.
- When turning his notes into a post, keep his facts and setup exactly as given. Don't reinterpret the setup.

## Git

Commit on a short-lived branch, fast-forward `main`, push, then delete the branch. Push only when asked.
