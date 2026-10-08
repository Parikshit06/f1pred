"""How the dashboard is drawn: CSS, SVG and colour.

Styled after a timing screen. Two rules hold everywhere: colour carries data
only (constructor colour identifies a car, the accent marks the model, green
and red mean better or worse than a baseline), and numbers are monospaced
and right-aligned so columns scan.

Self-contained - inline CSS, hand-built SVG, no CDN - so it renders the same
from Pages, a local file, or offline.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

# Constructor colours as (light surface, dark surface) pairs. Raw liveries fail
# contrast - Mercedes teal is 1.8:1 on white - so each team gets a lightness-
# shifted variant per surface on its own hue. The main four were checked for
# colour-vision deficiency; Ferrari and McLaren stay close for deuteranopes,
# which is why chart lines are also labelled at their ends.
TEAM_COLOURS = {
    "mercedes": ("#00917C", "#00A188"),
    "ferrari": ("#B00026", "#E04A66"),
    "mclaren": ("#B57500", "#BE8200"),
    "red_bull": ("#2F63AE", "#5288D4"),
    "aston_martin": ("#1C7A5A", "#2E9B75"),
    "alpine": ("#0A6E99", "#2C8FBF"),
    "williams": ("#2B7FB5", "#6BB2DE"),
    "rb": ("#4457C4", "#7C93F0"),
    "alphatauri": ("#3F6E88", "#6E9AB4"),
    "haas": ("#5C6469", "#98A2A9"),
    "audi": ("#0A7D22", "#2FA544"),
    "sauber": ("#1E8F30", "#46B657"),
    "alfa": ("#A02038", "#D1566C"),
    "cadillac": ("#8A6B12", "#B9932B"),
    "renault": ("#7A6E00", "#A89A1A"),
    "racing_point": ("#B44D82", "#DD7FAE"),
    "force_india": ("#B44D82", "#DD7FAE"),
    "toro_rosso": ("#2A6FD1", "#6B9BE8"),
}
FALLBACK_COLOUR = ("#646E7B", "#8A94A6")

TEAM_NAMES = {
    "mercedes": "Mercedes",
    "ferrari": "Ferrari",
    "red_bull": "Red Bull",
    "mclaren": "McLaren",
    "aston_martin": "Aston Martin",
    "alpine": "Alpine",
    "williams": "Williams",
    "rb": "RB",
    "alphatauri": "AlphaTauri",
    "haas": "Haas",
    "audi": "Audi",
    "sauber": "Sauber",
    "alfa": "Alfa Romeo",
    "cadillac": "Cadillac",
    "renault": "Renault",
    "racing_point": "Racing Point",
    "force_india": "Force India",
    "toro_rosso": "Toro Rosso",
}


def team_name(team: str | None) -> str:
    slug = (team or "").lower()
    return TEAM_NAMES.get(slug, (team or "").replace("_", " ").title())


def team_slug(team: str | None) -> str:
    slug = (team or "").lower()
    return slug if slug in TEAM_COLOURS else "default"


def team_colour(team: str | None) -> str:
    """A CSS reference, so the value follows the viewer's theme."""
    return f"var(--t-{team_slug(team)})"


def team_tokens(mode: int) -> str:
    """mode 0 = light value, 1 = dark value."""
    pairs = [f"--t-{k}:{v[mode]}" for k, v in TEAM_COLOURS.items()]
    pairs.append(f"--t-default:{FALLBACK_COLOUR[mode]}")
    return "; ".join(pairs) + ";"


def series_colour(s: dict) -> str:
    """Team colour, or a lighter tint of it for the second car in a garage."""
    base = team_colour(s.get("team"))
    return f"color-mix(in oklab, {base} 68%, var(--paper))" if s.get("second_car") else base


def esc(x) -> str:
    return html.escape(str(x))


def pct(x: float, dp: int = 1) -> str:
    """A probability as a percentage, rounded half up. Anything that rounds to
    zero is shown as below the smallest printable step rather than as 0, which
    reads as impossible, and anything short of certain never prints as 100%.

    Rounding is decided first and the bounds checked on the rounded value:
    Python's own formatting rounds halves to even, so 0.5% printed as 0%."""
    step = Decimal(1).scaleb(-dp)
    shown = Decimal(str(round(float(x) * 100, 9))).quantize(step, rounding=ROUND_HALF_UP)
    if shown <= 0:
        return f"&lt;{step}%"
    if x < 1 and shown >= 100:
        return f"&gt;{100 - step}%"
    return f"{shown}%"


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# Light is the FIA classification sheet: white stock, black figures, colour
# only on the constructor rule. Dark is the same document on the pit wall.
# Both are the subject's own vernacular, so neither is an afterthought.
CSS = """
*,*::before,*::after{box-sizing:border-box}
/* SVG ignores the hidden attribute without this. */
[hidden]{display:none!important}

:root{
  --paper:#fafaf7; --band:#f3f2ee; --card:#ffffff; --chip:#ebe9e3; --ink:#111113; --ink-2:#4b4b52;
  --ink-3:#6b6b72; --rule:#111113; --hair:#e0ded8; --bar-bg:#111113; --bar-ink:#fafaf7;
  --accent:#c8102e;            /* signal red: start lights and the live marker */
  --good:#12704a; --bad:#a8172e;
  --car-edge:rgba(16,16,18,.32);
  --grain:.030; --grain-blend:multiply;
  --mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:"Geist",ui-sans-serif,system-ui,-apple-system,sans-serif;
  --ease:cubic-bezier(.32,.72,0,1);
  __TEAM_LIGHT__
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0b0c0f; --band:#0f1115; --card:#14161b; --chip:#1e2128; --ink:#f4f4f1; --ink-2:#a8aab2;
    --ink-3:#80828a; --rule:#f4f4f1; --hair:#23262e; --bar-bg:#14161b; --bar-ink:#f4f4f1;
    --accent:#ff3040;
    --good:#3cc98a; --bad:#ff6b7a;
    --car-edge:rgba(255,255,255,.26);
    --grain:.055; --grain-blend:overlay;
    __TEAM_DARK__
  }
}
:root[data-theme="dark"]{
  --paper:#0b0c0f; --band:#0f1115; --card:#14161b; --chip:#1e2128; --ink:#f4f4f1; --ink-2:#a8aab2;
  --ink-3:#80828a; --rule:#f4f4f1; --hair:#23262e; --bar-bg:#14161b; --bar-ink:#f4f4f1;
  --accent:#ff3040;
  --good:#3cc98a; --bad:#ff6b7a;
  --car-edge:rgba(255,255,255,.26);
  --grain:.055; --grain-blend:overlay;
  __TEAM_DARK__
}

body{margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:15.5px; line-height:1.55;
  -webkit-font-smoothing:antialiased}

/* Faint noise so large plain areas don't look flat. */
body::before{content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
  opacity:var(--grain); mix-blend-mode:var(--grain-blend);
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)'/%3E%3C/svg%3E")}

.wrap{max-width:1140px; margin:0 auto; padding-inline:20px; padding-block:0 0}
h1,h2,h3{margin:0; text-wrap:balance; letter-spacing:-.02em; font-weight:700}
p{margin:0}
.mono{font-family:var(--mono); font-variant-numeric:tabular-nums}

/* ---- top bar: a floating island, detached from the edge ----------------- */
.bar{position:sticky; top:12px; z-index:20; max-width:1100px; margin:12px auto 0; padding-inline:12px}
.bar .inner{display:flex; flex-wrap:wrap; align-items:center; gap:6px 18px; min-height:48px;
  padding:6px 8px 6px 18px; border-radius:999px; background:color-mix(in srgb,var(--bar-bg) 88%,transparent);
  color:var(--bar-ink); -webkit-backdrop-filter:blur(16px) saturate(1.4); backdrop-filter:blur(16px) saturate(1.4);
  box-shadow:0 0 0 1px color-mix(in srgb,var(--bar-ink) 10%,transparent),
    0 18px 40px -22px rgba(10,10,12,.55); font-size:13px}
.brand{display:inline-flex; align-items:center; gap:10px; font-weight:650; letter-spacing:-.01em;
  color:var(--bar-ink); white-space:nowrap}
.state{display:inline-flex; gap:12px; align-items:baseline; opacity:.72; white-space:nowrap}
.state b{font-weight:500}
.state [data-countdown]{font-family:var(--mono); font-size:12px}
.bar .sep{flex:1}
.bar nav{display:flex; gap:4px}
/* The island inverts the page, so its links use the island's ink. */
.bar a{color:var(--bar-ink); text-decoration:none; opacity:.7; padding:7px 14px; border-radius:999px;
  transition:opacity .4s var(--ease), background .4s var(--ease)}
.bar a:hover{opacity:1}
.bar a.brand{opacity:1; padding:0; border-radius:0}
.bar nav a[aria-current]{opacity:1; background:color-mix(in srgb,var(--bar-ink) 12%,transparent)}
@media (max-width:640px){
  .bar{top:8px; margin-top:8px; padding-inline:8px}
  .bar .inner{padding-left:14px; gap:4px 10px}
  .state{display:none}
  .bar .long{display:none}
  .bar .lights{display:none}
  .bar a{padding:7px 8px}
  .bar nav{gap:0}
}
/* The smallest phones keep all three links on one line and drop a word. */
@media (max-width:370px){ .bar .mid{display:none} }
.lights{display:inline-flex; gap:3px; align-items:center; margin-right:4px}
.lights i{width:6px; height:6px; border-radius:50%;
  background:color-mix(in srgb, var(--bar-ink) 22%, transparent);
  animation:lights 3.2s ease-in-out 1 forwards; animation-delay:calc(var(--n) * 300ms)}
@keyframes lights{
  0%{background:color-mix(in srgb,var(--bar-ink) 22%,transparent)}
  9%,45%{background:#ff2d20; box-shadow:0 0 6px #ff2d2099}
  53%,100%{background:color-mix(in srgb,var(--bar-ink) 22%,transparent); box-shadow:none}
}

/* ---- masthead: race, one status line, the forecast --------------------- */
.mast{padding:64px 0 44px}
.mast h1{font-size:clamp(2.3rem,6vw,4rem); line-height:1; font-weight:700; letter-spacing:-.045em}
.kicker{display:inline-block; margin-bottom:16px; padding:4px 12px; border-radius:999px; background:var(--chip);
  font-size:12.5px; font-weight:500; color:var(--ink-2)}
.status{margin-top:16px; font-size:1.05rem; color:var(--ink-2)}
.status.lede{max-width:70ch}
.status b{color:var(--ink)}
.pill{display:inline-block; font-size:12px; font-weight:500;
  padding:3px 10px; border-radius:999px; background:var(--chip); color:var(--ink-2);
  vertical-align:2px; margin-right:6px; white-space:nowrap}
.pill.done{background:color-mix(in srgb,var(--good) 14%,var(--paper)); color:var(--good)}
.pill.live{background:color-mix(in srgb,var(--accent) 12%,var(--paper)); color:var(--accent)}
.meta{margin-top:6px; max-width:88ch; font-size:.84rem; color:var(--ink-3)}
/* What the site is, in one line, above the race name. */
.about{margin-bottom:22px; max-width:62ch; font-size:.95rem; line-height:1.5; color:var(--ink-2)}
.about b{color:var(--ink); font-weight:600}
/* An archived forecast from an earlier pipeline: a quiet footnote, not a warning. */
.legacy{margin-top:10px; padding-left:10px; border-left:2px solid var(--hair); max-width:78ch;
  font-size:.8rem; line-height:1.5; color:var(--ink-3)}

/* ---- sections: a plain heading, content under it ------------------------- */
section{padding-block:80px; border-top:2px solid var(--rule); position:relative; scroll-margin-top:64px}
/* Sections rise in once as they scroll into view (script adds .in). Hidden
   only when the script is running, so the page never depends on it. */
@media (max-width:640px){ section{padding-block:60px} }
.js main > section{opacity:0; transform:translateY(28px);
  transition:opacity .9s var(--ease), transform .9s var(--ease)}
.js main > section.in{opacity:1; transform:none}
section::before{content:""; position:absolute; inset:0; z-index:-1;
  left:50%; transform:translateX(-50%); width:100vw; max-width:100vw;
  background:var(--band); opacity:0}
section.band::before{opacity:1}
section > .lab{display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 12px; margin-bottom:12px}
section > .lab b{font-size:1.75rem; font-weight:700; letter-spacing:-.03em; color:var(--ink)}
section > .lab span{font-size:.85rem; color:var(--ink-3)}
section > .body{min-width:0}
.cap{color:var(--ink-2); font-size:1rem; max-width:64ch; margin-bottom:28px}
.scroll + .cap{margin-top:22px}
.cap a, footer a, .stages-note a, .odds-note a{color:var(--ink); text-decoration:underline;
  text-decoration-color:var(--ink-3); text-underline-offset:3px; text-decoration-thickness:1px}
.cap a:hover, footer a:hover{text-decoration-color:var(--accent)}

/* ---- data tables: rules, not cards ------------------------------------ */
.colhead{padding:0 0 7px; border-bottom:2px solid var(--rule);
  font-family:var(--sans); font-size:12px; font-weight:500; letter-spacing:0; color:var(--ink-3)}
.colhead span{text-align:right; line-height:1.25}
/* A heading that wraps on a phone sits on the rule like the rest. */
.colhead.gridC, .colhead.grid6, .colhead.gridq{align-items:end}
.colhead span:nth-child(3){text-align:left}
.row{padding:9px 0; border-bottom:1px solid var(--hair);
  transition:background .4s var(--ease), box-shadow .4s var(--ease);
  animation:slide .7s var(--ease) backwards;
  animation-delay:calc(var(--i,0) * 35ms + 60ms)}
/* Hovering a row shows its team colour in the margin. */
.row:hover{background:color-mix(in srgb,var(--tc,var(--ink)) 7%,transparent);
  box-shadow:inset 3px 0 0 var(--tc,var(--ink))}
.row:last-child{border-bottom:0}
@keyframes slide{from{opacity:0; transform:translateX(-6px)} to{opacity:1; transform:none}}
.pos{font-family:var(--mono); font-size:.92rem; font-weight:600; color:var(--ink-3);
  text-align:right; font-variant-numeric:tabular-nums}
.row.podium .pos{color:var(--ink)}
/* A hairline keeps a pale livery visible whichever surface it lands on. */
.car{width:3px; height:22px; border-radius:2px; background:var(--tc,var(--ink-3));
  box-shadow:inset 0 0 0 .5px var(--car-edge)}
.who{min-width:0}
.name{font-weight:600; font-size:.92rem; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; line-height:1.25}
.n-short{display:none}
@media (max-width:560px){ .n-full{display:none} .n-short{display:inline} }
.team{font-size:12px; color:var(--ink-3)}
.v{font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right;
  font-size:.86rem}
.v.lead{font-weight:600; font-size:.95rem}
.v.dim{color:var(--ink-3)}
/* 10th-90th percentile range under the projected total. */
.rng{display:block; font-size:9.5px; font-weight:400; color:var(--ink-3);
  letter-spacing:.02em; margin-top:1px}

.gridC{display:grid; grid-template-columns:22px 3px minmax(90px,1fr) 52px 66px 52px;
  align-items:center; gap:0 10px}

@media (max-width:480px){
  /* Too wide for a 320px screen with every column; drop "Now". */
  .gridC{grid-template-columns:20px 3px minmax(70px,1fr) 66px 50px; gap:0 8px}
  .gridC > .c-now, .colhead.gridC > span:nth-child(4){display:none}
  .wrap{padding-inline:14px}
  .bar .inner{padding:0 14px}
  /* The chart scrolls sideways rather than shrinking its labels to nothing. */
  .chart{overflow-x:auto}
  .chart svg{min-width:620px}
  p.chart-hint{display:block}
}

/* ---- two panels side by side (championship) --------------------------- */
.pair{display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:34px}
.panel h3{font-size:1.05rem; font-weight:650; letter-spacing:-.01em; color:var(--ink); margin-bottom:12px}

/* ---- track record: the winner's chance through each weekend ------------- */
.sub{font-size:1.05rem; font-weight:650; letter-spacing:-.01em; margin:34px 0 4px}
.sub + .cap{margin-bottom:18px}
.cap.tested{margin-top:24px}
.races{display:grid; border-top:2px solid var(--rule)}
.race{display:grid; grid-template-columns:minmax(200px,1fr) 2.6fr; gap:14px 32px; align-items:center;
  padding:18px 0 18px 18px; border-bottom:1px solid var(--hair); box-shadow:inset 3px 0 0 var(--tc,var(--ink))}
.race .rh b{display:block; font-size:1rem}
.race .rh a{color:inherit; text-decoration:underline; text-decoration-color:var(--ink-3);
  text-underline-offset:3px}
.race .rh a:hover{text-decoration-color:var(--ink)}
.racelist{margin:8px 0 72px}
.archived{margin-bottom:14px; font-size:.88rem; color:var(--ink-3)}
.archived a{color:var(--ink); text-underline-offset:3px; text-decoration-color:var(--ink-3)}
.race .rh span{font-size:.82rem; color:var(--ink-3)}
.weekend{display:grid; grid-template-columns:repeat(3,1fr); gap:10px 24px}
.wk{display:grid; gap:6px}
.wk .lbl{font-size:12px; font-weight:500; color:var(--ink-3)}
.wbar{display:block; height:6px; border-radius:3px; background:var(--chip); overflow:hidden}
.wbar i{display:block; height:100%; border-radius:3px; background:var(--tc,var(--ink));
  transform-origin:left; animation:grow .9s var(--ease) backwards}
.wv{display:flex; align-items:baseline; gap:8px}
.wv b{font-family:var(--mono); font-size:1rem; font-weight:600}
.wv small{font-size:.8rem; color:var(--ink-3)}
.wv i{font-style:normal; font-weight:700}
.wv i.ok{color:var(--good)}
.wv i.no{color:var(--bad)}
.wk.none .wv{font-size:.82rem; color:var(--ink-3)}
@media (max-width:700px){ .race{grid-template-columns:1fr} .weekend{gap:10px 14px} .wv small{display:none} }
/* ---- qualifying: the race board's columns, with pole in place of win ---- */
.gridq{display:grid; grid-template-columns:26px 3px minmax(120px,1.6fr) 120px 72px 72px;
  align-items:center; gap:0 12px}
.gridq.fin{grid-template-columns:26px 3px minmax(120px,1.6fr) 120px 72px 72px 60px}
@media (max-width:860px){
  .gridq{grid-template-columns:24px 3px minmax(96px,1fr) 96px 56px}
  .gridq.fin{grid-template-columns:24px 3px minmax(96px,1fr) 88px 52px 52px}
  .gridq > .c-top5{display:none}
}
@media (max-width:480px){
  .gridq{grid-template-columns:18px 3px minmax(70px,1fr) 70px 46px; gap:0 8px}
  .gridq.fin{grid-template-columns:18px 3px minmax(60px,1fr) 60px 40px 40px; gap:0 6px}
}

/* ---- plain tables ------------------------------------------------------ */
/* 10px gutter for the emphasis marker, outside the text margin. */
.scroll{overflow-x:auto; padding-left:10px; margin-left:-10px}
table{border-collapse:collapse; width:100%; font-size:.86rem}
/* Gaps between columns, not after them, so both edges sit on the margins. */
th,td{padding:8px 0; text-align:left; border-bottom:1px solid var(--hair);
  white-space:nowrap; vertical-align:baseline}
th+th,td+td{padding-left:26px}
th:first-child,td:first-child{padding-left:0}
thead th{font-family:var(--sans); font-size:12px; font-weight:500; letter-spacing:0; color:var(--ink-3);
  border-bottom:2px solid var(--rule)}
tbody tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right}
td.n{font-family:var(--mono); font-variant-numeric:tabular-nums}

/* ---- chart ------------------------------------------------------------- */
.chart{position:relative; width:100%}
.chart svg{display:block; width:100%; height:auto; overflow:visible}
.chart .gridline{stroke:var(--hair); stroke-width:1}
.chart .band{stroke:none; opacity:.13}
.chart .axis{stroke:var(--hair); stroke-width:1}
.chart text{font-family:var(--mono); font-size:10px; fill:var(--ink-3)}
.chart text.lbl{font-family:var(--sans); font-size:12px; font-weight:600; fill:var(--ink-2)}
.chart .lead{fill:none; stroke-width:1; opacity:.45; stroke-linejoin:round}
.chart .now{stroke:var(--accent); stroke-width:1; stroke-dasharray:2 3}
/* Lines draw in once, left to right; dash length is set in script. */
.chart .ser{fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round}
.chart .ser.draw{animation:draw 1.1s cubic-bezier(.35,.1,.25,1) backwards}
@keyframes draw{from{stroke-dashoffset:var(--len)} to{stroke-dashoffset:0}}
.chart .pt{opacity:0; animation:fadein .4s ease-out .9s forwards}
@keyframes fadein{to{opacity:1}}
.chart .proj{stroke-dasharray:5 4; opacity:.95}
.chart .pt{stroke:var(--paper); stroke-width:2}
.chart .hx{stroke:var(--ink-3); stroke-width:1}
.chart .hp{stroke:var(--paper); stroke-width:2}
.chart .hit{fill:transparent; cursor:crosshair}
.tip{position:absolute; pointer-events:none; z-index:5; min-width:132px;
  background:var(--paper); border:1px solid var(--rule); border-radius:4px;
  padding:8px 10px; box-shadow:0 8px 24px -10px rgba(0,0,0,.45);
  font-family:var(--mono); font-size:11px; line-height:1.5}
.tip .rd{font-family:var(--sans); color:var(--ink-3); font-size:11.5px; margin-bottom:5px}
.tip .r{display:flex; align-items:center; gap:7px; justify-content:space-between}
.tip .r i{width:8px; height:2px; border-radius:1px; flex:none}
.tip .r b{font-weight:600; margin-left:auto; font-variant-numeric:tabular-nums}
.tip .r em{font-style:normal; color:var(--ink-3); font-size:9.5px}
.chart-cap{font-size:.9rem; color:var(--ink-2); max-width:70ch; margin:34px 0 12px}
.chart-hint{display:none; font-size:.8rem; color:var(--ink-3); margin:0 0 8px}
.odds-note{font-size:.85rem; color:var(--ink-3); max-width:70ch; margin:18px 0 0}
.legend{display:flex; flex-wrap:wrap; gap:6px 16px; margin-top:14px;
  font-size:12.5px; color:var(--ink-2)}
.legend span{display:inline-flex; align-items:center; gap:6px}
.legend i{width:14px; height:3px; border-radius:1px}

/* ---- footer ------------------------------------------------------------ */
footer{border-top:1px solid var(--hair); margin-top:8px; padding:28px 0 64px;
  display:flex; flex-wrap:wrap; gap:8px 24px; justify-content:space-between;
  font-size:13px; color:var(--ink-3)}
footer a{color:var(--ink-2); white-space:nowrap}

/* ---- the race board: every column read off one distribution -------------- */
.grid6{display:grid;
  grid-template-columns:26px 3px minmax(120px,1.6fr) 72px 120px 64px 64px 56px;
  align-items:center; gap:0 12px}
.grid6.fin{grid-template-columns:26px 3px minmax(120px,1.6fr) 72px 120px 64px 64px 56px 60px}
.start small{display:block; font-size:9px; color:var(--ink-3); letter-spacing:.02em}
.v.fin{font-weight:600; color:var(--ink)}
.winc{display:flex; flex-direction:column; align-items:flex-end; gap:4px}
.mbar{display:block; width:100%; height:3px; margin-top:4px; border-radius:2px; background:var(--chip); overflow:hidden}
.mbar i{display:block; height:100%; background:color-mix(in srgb,var(--ink) 38%,transparent); border-radius:2px}
.pbar{display:block; width:100%; height:4px; border-radius:2px; background:var(--chip); overflow:hidden}
.pbar i{display:block; height:100%; border-radius:2px; background:var(--tc,var(--ink));
  transform-origin:left; animation:grow .7s cubic-bezier(.2,.7,.3,1) backwards;
  animation-delay:calc(var(--i,0) * 35ms + 120ms)}
@keyframes grow{from{transform:scaleX(0)}}

@media (max-width:860px){
  .grid6{grid-template-columns:24px 3px minmax(96px,1fr) 60px 96px 56px 48px}
  .grid6.fin{grid-template-columns:24px 3px minmax(96px,1fr) 56px 88px 52px 44px 52px}
  .grid6 > .c-top5{display:none}
}
@media (max-width:480px){
  .grid6{grid-template-columns:18px 3px minmax(70px,1fr) 42px 70px 46px 38px; gap:0 8px}
  .grid6.fin{grid-template-columns:18px 3px minmax(60px,1fr) 36px 60px 40px 34px 40px; gap:0 6px}
}

/* ---- the forecast: favourite, then the top five ----------------------- */
.hero{display:grid; grid-template-columns:minmax(200px,260px) 1fr; gap:28px 56px; align-items:center;
  margin-top:40px; padding:34px 0; border-block:1px solid var(--hair)}
@media (max-width:760px){ .hero{grid-template-columns:1fr} }
.fav .k, .t5head{font-family:var(--sans); font-size:12px; font-weight:500; letter-spacing:0; color:var(--ink-3)}
.fav .nm{font-size:1.6rem; font-weight:800; letter-spacing:-.02em; line-height:1.1; margin-top:8px}
.fav .tm{font-size:.85rem; color:var(--ink-3); margin-top:4px}
.fav .pc{font-family:var(--mono); font-size:3.4rem; font-weight:600; letter-spacing:-.04em; line-height:1;
  margin-top:18px; color:var(--tc,var(--ink)); font-variant-numeric:tabular-nums}
.fav .pcl{font-size:.85rem; color:var(--ink-3); margin-top:4px}
/* The straight: each car has come as far as its chance to win; the flag is 100%.
   A car's nose sits at 40px + (track - 52px) * p, so 100% touches the flag. */
.wins{min-width:0}
.wrow{display:grid; grid-template-columns:118px 1fr 48px; gap:14px; align-items:center; padding:5px 0}
.wins.fin .wrow{grid-template-columns:118px 1fr 48px 56px}
.whead{font-size:12px; font-weight:500; color:var(--ink-3); padding-bottom:6px}
.fr{text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums}
.wrow b.fr{font-weight:600}
.ffin{font-size:.85rem; color:var(--ink-2)}
.wname{font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  border-left:3px solid var(--tc,var(--ink)); padding-left:8px}
.wtrack{position:relative; height:26px; border-bottom:1px dashed var(--hair)}
.wtrack .gl{position:absolute; top:2px; bottom:0; width:1px; background:var(--hair);
  left:calc(40px + (100% - 52px) * var(--q))}
.wtrack .flag{position:absolute; right:0; top:0; bottom:-1px; width:10px; border-radius:1px;
  background:repeating-conic-gradient(var(--ink) 0 25%, var(--paper) 0 50%) 0 0/5px 5px; opacity:.85}
.carpos{position:absolute; bottom:3px; left:calc((100% - 52px) * var(--p)); display:flex;
  animation:drive 1.1s var(--ease) backwards; animation-delay:calc(var(--i) * 70ms + 150ms)}
.carsvg{width:40px; height:16px; fill:var(--tc,var(--ink)); flex:none}
.carsvg .w{fill:var(--ink-3)}
@keyframes drive{from{transform:translateX(-40px); opacity:0}}
.waxis{padding-top:2px}
.waxis .wtrack{height:14px; border:0}
.waxis .wtrack i{position:absolute; top:0; font-style:normal; font-family:var(--mono); font-size:10px;
  color:var(--ink-3); left:calc(40px + (100% - 52px) * var(--q)); transform:translateX(-50%)}
.fnote{font-size:.8rem; color:var(--ink-3); margin-top:8px}
@media (max-width:480px){
  .wrow{grid-template-columns:92px 1fr 36px; gap:8px}
  .wins.fin .wrow{grid-template-columns:88px 1fr 34px 36px; gap:6px}
  .wname{font-size:.84rem; padding-left:6px}
  .carsvg{width:32px; height:13px}
  .waxis .wtrack i:nth-child(even){display:none}
}

/* ---- forecast stage: one quiet line ----------------------------------- */
.stages{list-style:none; margin:30px 0 0; padding:0; display:flex; flex-wrap:wrap; gap:6px 0;
  font-size:.82rem; color:var(--ink-3)}
.stages li{display:inline-flex; align-items:center}
.stages li + li::before{content:""; width:28px; height:1px; background:var(--hair); margin:0 10px}
.stages li.done{color:var(--ink-2)}
.stages li.now{color:var(--ink); font-weight:600}
.stages li.now::after{content:""; width:6px; height:6px; border-radius:50%; background:var(--accent);
  margin-left:7px}
.stages-note{margin-top:6px; font-size:.82rem; color:var(--ink-3)}
.onpage{display:flex; flex-wrap:wrap; gap:6px 8px; margin-top:26px}
.onpage a{padding:6px 12px; border-radius:6px; background:var(--chip); color:var(--ink-2); font-size:.85rem;
  font-weight:500; text-decoration:none; transition:background .3s var(--ease), color .3s var(--ease)}
.onpage a:hover{color:var(--ink); background:color-mix(in srgb,var(--ink) 10%,var(--chip))}
@media (max-width:480px){ .stages li + li::before{width:12px; margin:0 6px} .stages{font-size:.78rem} }

/* ---- empty and notice states --------------------------------------------- */
.notice{border-left:3px solid var(--ink-3); padding:14px 18px;
  color:var(--ink-2); font-size:.9rem; max-width:64ch; background:var(--paper)}
.notice b{color:var(--ink)}

/* ---- method page ------------------------------------------------------------ */
.keyres td small{display:block; font-size:.74rem; color:var(--ink-3); font-weight:400}
.keyres tr.grp td{border-top:2px solid var(--hair)}
.keyres td{vertical-align:top}
.cap code{font-family:var(--mono); font-size:.82em;
  background:var(--chip); padding:1px 5px; border-radius:3px; color:var(--ink)}
/* One readable column: the method page is an article, not a dashboard. */
.article{max-width:740px; margin-inline:auto}
.article thead th{font-family:var(--sans); font-size:.8rem; letter-spacing:0; text-transform:none; color:var(--ink-3)}
.article .mast{padding:56px 0 30px}
.standfirst{margin-top:18px; font-size:1.2rem; line-height:1.55; color:var(--ink-2); max-width:58ch}
section.part{padding-block:40px 30px}
.part h2{font-size:1.5rem; letter-spacing:-.02em; margin-bottom:16px}
.part p, .part .cap{font-size:1rem; line-height:1.7; color:var(--ink-2); max-width:66ch; margin:0 0 16px}
.part p b, .part .cap b{color:var(--ink); font-weight:600}
.timeline{list-style:none; margin:26px 0 10px; padding:0; display:grid; grid-template-columns:repeat(4,1fr);
  gap:0 20px}
.timeline li{position:relative; padding-top:28px}
.timeline li::before{content:""; position:absolute; top:3px; left:0; width:12px; height:12px; border-radius:50%;
  background:var(--paper); border:2px solid var(--ink); z-index:1}
.timeline li::after{content:""; position:absolute; top:9px; left:18px; right:-14px; height:1px; background:var(--ink-3)}
.timeline li:last-child::before{background:var(--ink)}
.timeline li:last-child::after{display:none}
.timeline b{display:block; font-weight:600; margin-bottom:4px}
.timeline span{display:block; color:var(--ink-2); font-size:.9rem; line-height:1.5}
@media (max-width:640px){
  .timeline{grid-template-columns:1fr}
  .timeline li{padding:0 0 20px 28px}
  .timeline li::after{top:18px; bottom:-2px; left:5px; right:auto; width:1px; height:auto}
}
/* The explainer: chips, a hundred dots, three plain claims. */
.chips{list-style:none; margin:4px 0 22px; padding:0; display:flex; flex-wrap:wrap; gap:8px}
.chips li{padding:7px 14px; border-radius:999px; background:var(--chip); font-size:.92rem; color:var(--ink)}
.split{display:grid; grid-template-columns:1fr 220px; gap:24px 40px; align-items:center}
@media (max-width:640px){ .split{grid-template-columns:1fr} }
.dots{margin:0}
.dots div{display:grid; grid-template-columns:repeat(10,1fr); gap:6px; max-width:220px}
.dots i{aspect-ratio:1; border-radius:50%; background:var(--chip);
  animation:fadein .5s var(--ease) backwards; animation-delay:calc(var(--d,0) * 1ms)}
.dots i.on{background:var(--accent)}
.dots figcaption{margin-top:10px; font-family:var(--mono); font-size:12.5px; color:var(--ink-3)}
.example{margin-top:22px!important; padding:16px 0 0; border-top:1px solid var(--hair); color:var(--ink)!important}
.example b{font-family:var(--mono); font-weight:600}
.claims{list-style:none; margin:26px 0 8px; padding:0; counter-reset:c; display:grid; gap:22px}
.claims li{counter-increment:c; position:relative; padding-left:48px; line-height:1.6; color:var(--ink-2)}
.claims li::before{content:counter(c); position:absolute; left:0; top:-2px; width:32px; height:32px; border-radius:50%;
  display:grid; place-items:center; background:var(--ink); color:var(--paper); font-family:var(--mono);
  font-size:13px; font-weight:600}
.claims b{display:block; font-size:1.12rem; letter-spacing:-.01em; color:var(--ink); margin-bottom:2px}
.hood{margin:0 0 22px; display:grid; border-top:1px solid var(--hair)}
.hood > div{display:grid; grid-template-columns:150px 1fr; gap:4px 24px; padding:14px 0;
  border-bottom:1px solid var(--hair)}
.hood dt{font-weight:600; color:var(--ink)}
.hood dd{margin:0; color:var(--ink-2); line-height:1.55}
.hood dd small{display:block; margin-top:2px; font-family:var(--mono); font-size:12px; color:var(--ink-3)}
@media (max-width:560px){ .hood > div{grid-template-columns:1fr} }
.links{display:flex; flex-wrap:wrap; gap:8px 22px}
.links a{color:var(--ink); font-weight:500; text-underline-offset:3px; text-decoration-color:var(--ink-3)}
/* Method page: the key results before any detail. */
.kr-cap{font-size:.9rem; line-height:1.55; color:var(--ink-3); max-width:70ch; margin:0 0 14px}
.keyrow{display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:0 28px;
  border-top:2px solid var(--rule); margin-bottom:48px}
.kr{display:grid; gap:3px; align-content:start; padding:14px 0 16px; border-bottom:1px solid var(--hair)}
.kr-s{font-size:12px; color:var(--ink-3)}
.kr-m{font-size:.95rem; font-weight:600}
.kr-m small{display:block; font-family:var(--mono); font-size:11px; font-weight:400; color:var(--ink-3)}
.kr-v{margin-top:4px; font-family:var(--mono); font-size:1.4rem; font-weight:600; letter-spacing:-.02em}
.kr-v em{margin-left:8px; font-style:normal; font-size:.85rem; font-weight:400; color:var(--ink-3)}
.kr-r{font-size:12px; color:var(--ink-3)}
.kr-x{margin-top:4px; font-size:12.5px; font-weight:600}
.kr-x.good, .keyres .vd.good{color:var(--good)}
.kr-x.bad, .keyres .vd.bad{color:var(--bad)}
.kr-x.flat, .keyres .vd.flat{color:var(--ink-2); font-weight:400}
.keyres td small{display:block}
.keyres{font-size:.84rem}
.keyres th+th, .keyres td+td{padding-left:18px}
.keyres tr.grp td{padding-top:22px; border-top:0; border-bottom:2px solid var(--rule)}
.keyres tr.grp td small{display:inline; margin-left:6px}
.keyres .vd{white-space:nowrap; font-weight:600}

/* Calibration: three small reliability diagrams. */
.calgrid{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px 26px; margin-top:8px}
@media (max-width:640px){ .calgrid{grid-template-columns:minmax(0,300px)} }
.calp{margin:0}
.calp figcaption{font-weight:600; font-size:.95rem; margin-bottom:6px}
.calp svg{display:block; width:100%; height:auto; overflow:visible}
.calp text{font-family:var(--mono); font-size:9.5px; fill:var(--ink-3)}
.calp text.ax{font-family:var(--sans); font-size:10.5px}
.calp .gridline{stroke:var(--hair); stroke-width:1}
.calp .diag{stroke:var(--ink-3); stroke-width:1.25; stroke-dasharray:4 4}
.calp .ci{stroke:var(--ink-3); stroke-width:1.5; stroke-linecap:round; opacity:.7}
.calp .dot{fill:var(--ink); stroke:var(--paper); stroke-width:2}
.calp .off .dot{fill:var(--bad)}
.calp .hit{fill:transparent}
.calp .cpt:hover .dot{r:6}
.cal-key{display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:14px; font-size:12px; color:var(--ink-3)}
.cal-key span{display:inline-flex; align-items:center; gap:6px}
.cal-key i{display:inline-block}
.k-diag{width:16px; border-top:1.5px dashed var(--ink-3)}
.k-dot, .k-off{width:9px; height:9px; border-radius:50%; background:var(--ink)}
.k-off{background:var(--bad)}
.k-ci{width:1.5px; height:12px; background:var(--ink-3)}

@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important; transition:none!important}
}
"""

CSS = CSS.replace("__TEAM_LIGHT__", team_tokens(0)).replace("__TEAM_DARK__", team_tokens(1))

# The mark: the five start lights, on the island's ink.
ICON = (
    "<link rel='icon' href=\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='9' fill='%23111113'/%3E"
    + "".join(f"%3Ccircle cx='{6 + 5 * i}' cy='16' r='2' fill='%23ff2d20'/%3E" for i in range(5))
    + '%3C/svg%3E">'
)

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Geist:wght@400;500;600;700;800&"
    'family=Geist+Mono:wght@400;500;600&display=swap">'
)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def _row_head(i: int, r: dict) -> str:
    """Position, constructor colour, driver.

    Both spellings of the name ship in the markup and CSS picks one. Truncating
    a driver with an ellipsis on a phone looks unfinished, and a surname is what
    a timing screen shows anyway.
    """
    full = esc(r.get("name", ""))
    short = esc(r.get("short") or r.get("name", ""))
    return (
        f"<div class='pos'>{i + 1}</div><div class='car'></div>"
        f"<div class='who'><div class='name'>"
        f"<span class='n-full'>{full}</span><span class='n-short'>{short}</span></div>"
        f"<div class='team'>{esc(team_name(r.get('team')))}</div></div>"
    )


def _start_cell(r: dict) -> str:
    """Where the car starts, and where it qualified when a penalty moved it."""
    grid, quali = r.get("grid"), r.get("quali_position")
    if not isinstance(grid, (int, float)):
        return "<div class='v dim start'>&mdash;</div>"
    moved = isinstance(quali, (int, float)) and int(quali) != int(grid)
    note = f"<small>Q{int(quali)}</small>" if moved else ""
    return f"<div class='v dim start'>P{int(grid)}{note}</div>"


def race_board(rows: list[dict], grid_known: bool = True, shown: int = 10) -> str:
    """The race forecast's top ten, all read off the same finishing
    distribution so win <= podium <= top 5 always holds."""
    finished = any(isinstance(r.get("finished"), (int, float)) for r in rows)
    cls = "grid6 fin" if finished else "grid6"
    head = (
        f"<div class='colhead {cls}'><span>#</span><span></span><span>Driver</span>"
        f"<span>{'Grid' if grid_known else 'Proj. grid'}</span><span>Win</span><span>Podium</span>"
        "<span class='c-top5'>Top 5</span><span>Top 10</span>"
        + ("<span>Result</span>" if finished else "")
        + "</div>"
    )

    def one(i: int, r: dict) -> str:
        p = float(r.get("p_win") or 0)
        fin = r.get("finished")
        return (
            f"<div class='row {cls}{' podium' if i < 3 else ''}' "
            f'style="--i:{min(i, 12)}; --tc:{team_colour(r.get("team"))}">'
            + _row_head(i, r)
            + _start_cell(r)
            + f"<div class='v lead winc'><span data-count>{pct(p)}</span>"
            + f"<span class='pbar'><i style='width:{min(p, 1.0) * 100:.1f}%'></i></span></div>"
            + f"<div class='v'>{pct(r.get('p_podium') or 0, 0)}"
            + f"<span class='mbar'><i style='width:{min(float(r.get('p_podium') or 0), 1.0) * 100:.0f}%'></i></span></div>"
            + f"<div class='v dim c-top5'>{pct(r.get('p_top5') or 0, 0)}</div>"
            + f"<div class='v dim'>{pct(r.get('p_top10') or 0, 0)}</div>"
            + (
                (
                    f"<div class='v fin'>P{int(fin)}</div>"
                    if isinstance(fin, (int, float))
                    else "<div class='v dim'>&mdash;</div>"
                )
                if finished
                else ""
            )
            + "</div>"
        )

    out = ["<div class='board'>", head]
    out += [one(i, r) for i, r in enumerate(rows[:shown])]
    out.append("</div>")
    return "".join(out)


CAR = (
    "<svg class='carsvg' viewBox='0 0 44 18' aria-hidden='true'>"
    "<g class='w'><rect x='7' y='0' width='7' height='4' rx='1'/><rect x='7' y='14' width='7' height='4' rx='1'/>"
    "<rect x='30' y='0' width='6' height='4' rx='1'/><rect x='30' y='14' width='6' height='4' rx='1'/></g>"
    "<path d='M1 6.5h7l5-2h15l7 2.5h8v2h-8l-7 2.5H13l-5-2H1z'/></svg>"
)


def win_track(rows: list[dict], finished: bool = False, n: int = 8) -> str:
    """The contenders as cars on a straight, in the table's order: how far a
    car has come is its chance to win, and the chequered flag is 100%."""
    lead = rows[:n]
    if not lead:
        return ""
    grid = "".join(f"<i class='gl' style='--q:{q}'></i>" for q in (0.25, 0.5, 0.75))
    out = [
        f"<div class='wins{' fin' if finished else ''}'>"
        "<div class='wrow whead'><span></span><span>Chance to win</span>"
        "<span></span>" + ("<span class='fr'>Result</span>" if finished else "") + "</div>"
    ]
    for i, r in enumerate(lead):
        p = min(max(float(r.get("p_win") or 0), 0.0), 1.0)
        fin = r.get("finished")
        out.append(
            f"<div class='wrow' style='--tc:{team_colour(r.get('team'))}; --p:{p:.3f}; --i:{i}'>"
            f"<span class='wname'>{esc(r.get('short') or r.get('name', ''))}</span>"
            f"<span class='wtrack'>{grid}<i class='flag'></i><span class='carpos'>{CAR}</span></span>"
            f"<b class='fr'>{pct(p, 0)}</b>"
            + (
                (
                    f"<span class='fr ffin'>P{int(fin)}</span>"
                    if isinstance(fin, (int, float))
                    else "<span class='fr ffin'>&mdash;</span>"
                )
                if finished
                else ""
            )
            + "</div>"
        )
    ticks = "".join(f"<i style='--q:{q}'>{q:.0%}</i>" for q in (0, 0.25, 0.5, 0.75, 1))
    out.append(
        f"<div class='wrow waxis'><span></span><span class='wtrack'>{ticks}</span></div>"
        "<p class='fnote'>Each percentage is the share of 10,000 simulated races that driver won. "
        "The nearer a car is to the chequered flag, the likelier the win.</p></div>"
    )
    return "".join(out)


STAGES = [
    ("pre_practice", "Before practice"),
    ("pre_quali", "After practice"),
    ("post_quali", "After qualifying"),
    ("result", "Race result"),
]


def stage_track(current: str) -> str:
    """Where this forecast sits in the weekend, as one quiet line."""
    keys = [k for k, _ in STAGES]
    at = keys.index(current) if current in keys else 0
    items = "".join(
        f"<li class='{'now' if i == at else ('done' if i < at else '')}'>{name}</li>"
        for i, (_, name) in enumerate(STAGES)
    )
    return (
        f"<ol class='stages' aria-label='Forecast stage'>{items}</ol>"
        "<p class='stages-note'>A new forecast is logged at each step of the weekend and "
        "graded once the race is run. <a href='method.html'>How it works</a></p>"
    )


def quali_board(rows: list[dict], qualified: bool = False, shown: int = 10) -> str:
    """The qualifying forecast's top ten, in expected order. Once qualifying
    has run, a last column shows where each driver starts.

    The middle column is the top-5 chance. Forecasts logged before
    2026-09-30 filled their "podium" field with it, so reading top 5 keeps
    every logged forecast labelled correctly without rewriting any of them.
    """
    if rows and all(isinstance(r.get("exp_position"), (int, float)) for r in rows):
        rows = sorted(rows, key=lambda r: r["exp_position"])
    cls = "gridq fin" if qualified else "gridq"

    def one(i: int, r: dict) -> str:
        p = float(r.get("p_win") or 0)
        top5 = float(r.get("p_top5") or 0)
        g = r.get("grid")
        return (
            f"<div class='row {cls}{' podium' if i < 3 else ''}' "
            f'style="--i:{min(i, 12)}; --tc:{team_colour(r.get("team"))}">'
            + _row_head(i, r)
            + f"<div class='v lead winc'><span data-count>{pct(p)}</span>"
            + f"<span class='pbar'><i style='width:{min(p, 1.0) * 100:.1f}%'></i></span></div>"
            + f"<div class='v c-top5'>{pct(top5, 0)}"
            + f"<span class='mbar'><i style='width:{min(top5, 1.0) * 100:.0f}%'></i></span></div>"
            + f"<div class='v dim'>{pct(r.get('p_top10') or 0, 0)}</div>"
            + (
                (
                    f"<div class='v fin'>P{int(g)}</div>"
                    if isinstance(g, (int, float))
                    else "<div class='v dim'>&mdash;</div>"
                )
                if qualified
                else ""
            )
            + "</div>"
        )

    out = [
        f"<div class='board'><div class='colhead {cls}'><span>#</span><span></span><span>Driver</span>"
        "<span>Pole</span><span class='c-top5'>Top 5</span><span>Top 10</span>"
        + ("<span>Starts</span>" if qualified else "")
        + "</div>"
    ]
    out += [one(i, r) for i, r in enumerate(rows[:shown])]
    return "".join(out) + "</div>"


def _odds(p: float, alive: bool = True) -> str:
    """Title odds, held between <0.1% and 99.9%.

    10,000 simulated seasons can't resolve 1 in 10,000, so a driver who can
    still win it mathematically is never shown at 0 or 100. allocate_odds only
    hands out 100% once the title is settled.
    """
    if not alive:
        return "<span title='can no longer win the title'>out</span>"
    if p >= 1.0:
        return "clinched"
    if p < 0.0005:
        return "&lt;0.1%"
    return f"{min(p, 0.999) * 100:.1f}%"


def championship_panel(title: str, rows: list[dict], name_key: str, sub_key: str | None) -> str:
    out = [
        f"<div class='panel'><h3>{esc(title)}</h3>",
        (
            "<div class='colhead gridC'><span></span><span></span><span>"
            f"{'Driver' if sub_key else 'Constructor'}</span>"
            "<span>Now</span><span>Projected</span><span>Title</span></div>"
        ),
    ]
    for i, r in enumerate(rows):
        out.append(
            f"<div class='row gridC{' podium' if i < 3 else ''}' "
            f'style="--i:{i}; --tc:{team_colour(r.get("team"))}">'
            f"<div class='pos'>{i + 1}</div><div class='car'></div>"
            f"<div class='who'><div class='name'>"
            f"{esc(team_name(r[name_key]) if not sub_key else r[name_key])}</div>"
            + (f"<div class='team'>{esc(team_name(r.get('team')))}</div>" if sub_key else "")
            + "</div>"
            + f"<div class='v dim c-now'>{r['now']:.0f}</div>"
            + f"<div class='v lead'>{r['projected']:.0f}"
            + (
                f"<span class='rng'>{r['low']:.0f}\u2013{r['high']:.0f}</span>"
                if r.get("low") is not None
                else ""
            )
            + "</div>"
            + f"<div class='v'>{_odds(r['p_title'], r.get('alive', True))}"
            + f"<span class='mbar'><i style='width:{min(float(r['p_title']), 1.0) * 100:.0f}%'></i></span></div>"
            + "</div>"
        )
    return "".join(out) + "</div>"


# ---------------------------------------------------------------------------
# Points progression chart
# ---------------------------------------------------------------------------
def progression_chart(
    series: list[dict],
    last_actual_round: int,
    width: int = 880,
    height: int = 330,
) -> str:
    """Cumulative points by round: solid for what happened, dashed for projection.

    Dashes mean projected and nothing else. The second car in a garage is a
    lighter tint of the team colour, and every line is labelled at its end.
    """
    if not series:
        return ""

    # Round numbers arrive as strings when the series has been through JSON,
    # which is the normal path: the page is rendered from a logged prediction
    # rather than from the objects that produced it.
    def _ints(d):
        return {int(k): float(v) for k, v in (d or {}).items()}

    series = [
        {**s_, "points": _ints(s_["points"]), "low": _ints(s_.get("low")), "high": _ints(s_.get("high"))}
        for s_ in series
        if s_.get("points")
    ]
    if not series:
        return ""

    pad_l, pad_r, pad_t, pad_b = 44, 132, 16, 30
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    rounds = sorted({r for s in series for r in s["points"]})
    x_min, x_max = min(rounds), max(rounds)
    y_max = max(max(list(s["points"].values()) + list((s.get("high") or {}).values()) or [0]) for s in series)
    step = 100 if y_max > 320 else 50
    y_top = (int(y_max // step) + 1) * step

    def X(r: float) -> float:
        return pad_l + (r - x_min) / max(x_max - x_min, 1) * plot_w

    def Y(v: float) -> float:
        return pad_t + plot_h - (v / y_top) * plot_h

    out = [
        (
            f"<div class='chart'><svg viewBox='0 0 {width} {height}' role='img' "
            f"aria-label='Cumulative championship points by round, actual then projected'>"
        )
    ]

    for v in range(0, y_top + 1, step):
        out.append(
            f"<line class='gridline' x1='{pad_l}' y1='{Y(v):.1f}' x2='{pad_l + plot_w}' y2='{Y(v):.1f}'/>"
        )
        out.append(f"<text x='{pad_l - 8}' y='{Y(v) + 3.5:.1f}' text-anchor='end'>{v}</text>")

    for r in rounds:
        if r % 3 == 0 or r == x_min or r == x_max:
            out.append(f"<text x='{X(r):.1f}' y='{pad_t + plot_h + 18:.0f}' text-anchor='middle'>R{r}</text>")

    nx = X(last_actual_round)
    out.append(f"<line class='now' x1='{nx:.1f}' y1='{pad_t}' x2='{nx:.1f}' y2='{pad_t + plot_h}'/>")
    out.append(f"<text x='{nx + 5:.1f}' y='{pad_t + 10}' text-anchor='start'>projected &rarr;</text>")

    # End labels sit near their line's final value. Two passes: push down to a
    # minimum gap, then if the stack overruns the plot, shift it up and re-space
    # upward. Labels moved more than a few pixels get a leader line.
    ends = sorted(((s["points"][max(s["points"])], s) for s in series), key=lambda t: -t[0])
    # Each end label is two lines - name over value - so 26 left them touching.
    gap = 31.0
    wanted = [Y(value) for value, _ in ends]
    placed = list(wanted)
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    overflow = placed[-1] - (pad_t + plot_h - 10) if placed else 0
    if overflow > 0:
        placed = [y - overflow for y in placed]
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - gap)
        placed = [max(y, pad_t + 6) for y in placed]

    for (value, s), want, y in zip(ends, wanted, placed):
        colour = series_colour(s)
        actual = [(r, v) for r, v in sorted(s["points"].items()) if r <= last_actual_round]
        future = [(r, v) for r, v in sorted(s["points"].items()) if r >= last_actual_round]

        def path(pts):
            return "M" + " L".join(f"{X(r):.1f} {Y(v):.1f}" for r, v in pts)

        # The projected range, drawn behind its line. Without it the mean reads
        # as a promise; with it a leader who retires twice is visibly inside
        # the ordinary spread rather than an upset.
        lo, hi = s.get("low") or {}, s.get("high") or {}
        band = [r for r, _ in future if r in lo and r in hi]
        if len(band) > 1:
            top = " L".join(f"{X(r):.1f} {Y(hi[r]):.1f}" for r in band)
            bottom = " L".join(f"{X(r):.1f} {Y(lo[r]):.1f}" for r in reversed(band))
            out.append(f"<path class='band' d='M{top} L{bottom} Z' fill='{colour}'/>")

        if actual:
            out.append(f"<path class='ser' d='{path(actual)}' stroke='{colour}'/>")
        if len(future) > 1:
            out.append(f"<path class='ser proj' d='{path(future)}' stroke='{colour}'/>")
        if actual:
            r, v = actual[-1]
            out.append(f"<circle class='pt' cx='{X(r):.1f}' cy='{Y(v):.1f}' r='3.5' fill='{colour}'/>")

        # A label that had to move gets a hairline back to its own line end.
        if abs(y - want) > 2:
            x0 = pad_l + plot_w
            out.append(
                f"<path class='lead' d='M{x0:.1f} {want:.1f} "
                f"L{x0 + 6:.1f} {want:.1f} "
                f"L{x0 + 18:.1f} {y:.1f} "
                f"L{x0 + 24:.1f} {y:.1f}' stroke='{colour}'/>"
            )
        out.append(
            f"<text class='lbl' x='{pad_l + plot_w + 28}' y='{y + 3.5:.1f}' text-anchor='start'>"
            f"{esc(s['label'])}</text>"
            f"<text x='{pad_l + plot_w + 28}' y='{y + 15:.1f}' text-anchor='start'>{value:.0f}</text>"
        )

    out.append(
        f"<line class='axis' x1='{pad_l}' y1='{pad_t + plot_h}' x2='{pad_l + plot_w}' y2='{pad_t + plot_h}'/>"
    )

    # Hover furniture. Hidden until the pointer is over the plot; the script
    # moves it rather than re-rendering, so there is nothing to lay out twice.
    out.append(f"<rect class='hit' x='{pad_l}' y='{pad_t}' width='{plot_w}' height='{plot_h}'/>")
    out.append(
        f"<g class='hover' hidden>"
        f"<line class='hx' y1='{pad_t}' y2='{pad_t + plot_h}'/>"
        + "".join(f"<circle class='hp' r='4.5' data-s='{i}'/>" for i in range(len(series)))
        + "</g>"
    )
    out.append("</svg>")
    out.append("<div class='tip' hidden></div>")

    legend = "".join(
        f"<span><i style='background:{series_colour(s)}'></i>{esc(s['label'])}</span>" for s in series
    )
    out.append(f"<div class='legend'>{legend}</div>")

    payload = {
        "padL": pad_l,
        "padT": pad_t,
        "plotW": plot_w,
        "plotH": plot_h,
        "width": width,
        "height": height,
        "xMin": x_min,
        "xMax": x_max,
        "yTop": y_top,
        "lastActual": last_actual_round,
        "series": [
            {
                "label": s["label"],
                "colour": series_colour(s),
                "points": {str(r): round(v, 1) for r, v in sorted(s["points"].items())},
            }
            for s in series
        ],
    }
    out.append(f"<script type='application/json' class='chart-data'>{json.dumps(payload)}</script>")
    out.append("</div>")
    return "".join(out)


WEEKEND_STEPS = ("Before practice", "After practice", "After qualifying")


def _step_label(step: dict) -> str:
    if step["stage"] == "post_quali":
        return "After qualifying"
    return "After practice" if step.get("note") == "with this weekend's practice" else "Before practice"


def record_table(history: list[dict]) -> str:
    """One row per graded race: the chance the forecast gave the eventual
    winner at each step of the weekend, and whether its pick was right."""
    rows = []
    for r in history:
        last = {_step_label(st): st for st in r["steps"]}
        cells = []
        for label in WEEKEND_STEPS:
            st = last.get(label)
            if not st:
                cells.append(
                    f"<div class='wk none'><span class='lbl'>{label}</span>"
                    "<span class='wv'>not available for this race</span></div>"
                )
                continue
            p, ok = float(st["p_winner"]), st["hit"]
            cells.append(
                f"<div class='wk'><span class='lbl'>{label}</span>"
                f"<span class='wbar'><i style='width:{min(p, 1.0) * 100:.1f}%'></i></span>"
                f"<span class='wv'><b>{pct(p, 0)}</b><small>pick {esc(st['pick'])} "
                f"<i class='{'ok' if ok else 'no'}'>{'&#10003;' if ok else '&#10007;'}</i></small></span></div>"
            )
        rows.append(
            f"<article class='race' style='--tc:{team_colour(r.get('winner_team'))}'>"
            f"<div class='rh'><b><a href='race-{int(r['season'])}-{int(r['round']):02d}.html'>{esc(r['race'])}</a></b>"
            f"<span>{r['season']} round {r['round']} &middot; won by {esc(r['winner'])}</span></div>"
            "<div class='weekend'>" + "".join(cells) + "</div></article>"
        )
    return "<div class='races'>" + "".join(rows) + "</div>"


STAGE_LABELS = {
    "pre_practice": "Before practice",
    "pre_quali": "After practice",
    "post_quali": "After qualifying",
    "result": "Race finished",
}


def top_bar(
    prediction: dict | None, generated: datetime, stage: str | None = None, page: str = "forecast"
) -> str:
    """A floating island: the wordmark and forecast state, then navigation."""
    lights = "<span class='lights'>" + "".join(f"<i style='--n:{n}'></i>" for n in range(5)) + "</span>"
    name = "F1<span class='mid'> Prediction</span><span class='long'> System</span>"
    bits = [f"<a class='brand' href='index.html'>{lights}<span>{name}</span></a>"]
    if prediction and stage:
        state = [f"<b>{STAGE_LABELS.get(stage, stage)}</b>"]
        start = prediction.get("race_start_utc")
        if start and stage != "result":
            state.append(f"<span data-countdown='{esc(start)}'>&mdash;</span>")
        bits.append(f"<span class='state'>{''.join(state)}</span>")
    bits.append("<span class='sep'></span>")
    nav = [
        ("forecast", "index.html", "Forecast"),
        ("races", "races.html", "Past races"),
        ("method", "method.html", "Method<span class='long'> &amp; accuracy</span>"),
    ]
    links = []
    for key, href, label in nav:
        current = " aria-current='page'" if key == page else ""
        links.append(f"<a href='{href}'{current}>{label}</a>")
    bits.append("<nav>" + "".join(links) + "</nav>")
    return f"<div class='bar'><div class='inner'>{''.join(bits)}</div></div>"


# The page's only script: sections easing in, the countdown to the race, the
# headline numbers counting up once, and reading values off the chart. The
# page is complete without it.
SCRIPT = r"""
(function(){
  // Sections rise in as they reach the viewport; everything shows at once
  // where the observer is missing or motion is reduced.
  var secs = document.querySelectorAll('main > section');
  var still = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  if(!('IntersectionObserver' in window) || still){
    secs.forEach(function(s){ s.classList.add('in'); });
  } else {
    var io = new IntersectionObserver(function(es){
      es.forEach(function(e){ if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); } });
    }, {rootMargin:'0px 0px -8% 0px'});
    secs.forEach(function(s){ io.observe(s); });
  }
})();
(function(){
  function counts(el){
    var t = Date.parse(el.dataset.countdown); if(isNaN(t)) return;
    var ms = t - Date.now();
    if(ms <= 0){ el.textContent = 'race under way'; return; }
    var d = Math.floor(ms/86400000), h = Math.floor(ms%86400000/3600000), m = Math.floor(ms%3600000/60000);
    el.textContent = 'lights out in ' + (d ? d+'d ' : '') + h + 'h ' + m + 'm';
  }
  function tick(){
    document.querySelectorAll('[data-countdown]').forEach(counts);
  }
  tick(); setInterval(tick, 30000);

  var still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Headline probabilities count up once. Only cells tagged data-count are
  // touched, because the animation rewrites textContent and would otherwise
  // destroy the range span inside the championship totals. The final value is
  // already in the markup, so the page is right before the script runs and
  // right if it never does.
  if(!still){
    document.querySelectorAll('[data-count]').forEach(function(el, i){
      var end = parseFloat(el.textContent);
      if(isNaN(end)) return;
      var suffix = el.textContent.replace(/[\d.]+/, ''), t0 = null, dur = 620, delay = 90 + i * 35;
      function step(ts){
        if(t0 === null) t0 = ts;
        var k = Math.min(Math.max(ts - t0 - delay, 0) / dur, 1);
        el.textContent = (end * (1 - Math.pow(1 - k, 3))).toFixed(1) + suffix;
        if(k < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
  }

  // Line draw-in: the dash length has to come from the rendered path, so it is
  // measured here rather than guessed in CSS.
  document.querySelectorAll('.chart .ser').forEach(function(path, i){
    if(still) return;
    var len = path.getTotalLength();
    if(!len) return;
    path.style.setProperty('--len', len);
    path.style.strokeDasharray = path.classList.contains('proj') ? '5 4' : len + ' ' + len;
    if(!path.classList.contains('proj')){
      path.style.animationDelay = (i * 60) + 'ms';
      path.classList.add('draw');
    }
  });

  // ---- chart hover -------------------------------------------------------
  document.querySelectorAll('.chart').forEach(function(chart){
    var raw = chart.querySelector('.chart-data');
    var svg = chart.querySelector('svg');
    var hit = chart.querySelector('.hit');
    var grp = chart.querySelector('.hover');
    var tip = chart.querySelector('.tip');
    if(!raw || !svg || !hit || !grp || !tip) return;

    var d; try { d = JSON.parse(raw.textContent); } catch(e){ return; }
    var dots = grp.querySelectorAll('.hp');
    var line = grp.querySelector('.hx');
    var rounds = [];
    for(var r = d.xMin; r <= d.xMax; r++) rounds.push(r);

    function X(r){ return d.padL + (r - d.xMin) / Math.max(d.xMax - d.xMin, 1) * d.plotW; }
    function Y(v){ return d.padT + d.plotH - (v / d.yTop) * d.plotH; }

    function show(evt){
      // The SVG scales to its container, so pointer pixels have to be mapped
      // back into viewBox units before they mean anything.
      var box = svg.getBoundingClientRect();
      var vx = (evt.clientX - box.left) / box.width * d.width;
      var near = rounds[0], best = Infinity;
      rounds.forEach(function(r){
        var gap = Math.abs(X(r) - vx);
        if(gap < best){ best = gap; near = r; }
      });

      var rows = '', any = false;
      d.series.forEach(function(s, i){
        var v = s.points[String(near)];
        var dot = dots[i];
        if(v === undefined){ dot.setAttribute('r', 0); return; }
        any = true;
        dot.setAttribute('r', 4.5);
        dot.setAttribute('cx', X(near).toFixed(1));
        dot.setAttribute('cy', Y(v).toFixed(1));
        dot.setAttribute('fill', s.colour);
        rows += '<div class="r"><i style="background:' + s.colour + '"></i>' +
                '<span>' + s.label + '</span><b>' + Math.round(v) + '</b></div>';
      });
      if(!any) return;

      grp.hidden = false;
      line.setAttribute('x1', X(near).toFixed(1));
      line.setAttribute('x2', X(near).toFixed(1));

      var tail = near > d.lastActual ? '<em> projected</em>' : '';
      tip.innerHTML = '<div class="rd">Round ' + near + tail + '</div>' + rows;
      tip.hidden = false;

      // Keep the tooltip inside the chart, flipping side near the right edge.
      var px = X(near) / d.width * box.width;
      var w = tip.offsetWidth || 150;
      tip.style.left = Math.max(0, Math.min(px + 14, box.width - w - 4)) + 'px';
      tip.style.top = Math.max(0, (evt.clientY - box.top) - tip.offsetHeight - 12) + 'px';
    }

    function hide(){ grp.hidden = true; tip.hidden = true; }

    hit.addEventListener('pointermove', show);
    hit.addEventListener('pointerdown', show);
    hit.addEventListener('pointerleave', hide);
  });
})();
"""


BRAND = "F1 Prediction System"
DESCRIPTION = (
    "Probabilistic forecasts for every Formula 1 Grand Prix: race, qualifying and championship "
    "chances, published before each session and graded after the race."
)


def footer(repo_url: str) -> str:
    """Where the data comes from and where the code lives, and nothing else."""
    return (
        "<footer><span>Unofficial, not associated with Formula 1. Data from "
        "<a href='https://github.com/jolpica/jolpica-f1'>jolpica-f1</a> and "
        "<a href='https://openf1.org'>OpenF1</a>.</span>"
        f"<span><a href='{esc(repo_url)}'>Source code on GitHub</a></span></footer>"
    )


def document(body: str, title: str = BRAND) -> str:
    """A complete, self-contained HTML page."""
    title = title if title == BRAND else f"{title} \u00b7 {BRAND}"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1,viewport-fit=cover'>"
        f"<meta name='description' content='{esc(DESCRIPTION)}'>"
        f"<title>{esc(title)}</title>{ICON}{FONTS}<style>{CSS}</style>"
        "<script>document.documentElement.classList.add('js')</script></head>"
        f"<body>{body}<script>{SCRIPT}</script></body></html>"
    )


def utcnow() -> datetime:
    return datetime.now(UTC)


def _calibration_panel(title: str, rows: list[dict], min_n: int = 10) -> str:
    """One reliability diagram: what the forecast said against how often it
    happened, a dashed diagonal for perfect calibration, and each band's 95%
    range as a whisker. Bands with fewer than `min_n` cases are left out."""
    rows = [r for r in rows or [] if int(r.get("n", 0)) >= min_n]
    if not rows:
        return ""
    size, pl, pb, pt, pr = 220, 34, 30, 8, 10
    w, h = size - pl - pr, size - pt - pb

    def x(v: float) -> float:
        return pl + v * w

    def y(v: float) -> float:
        return pt + (1 - v) * h

    out = [
        (
            f"<figure class='calp'><figcaption>{esc(title)}</figcaption>"
            f"<svg viewBox='0 0 {size} {size}' role='img' aria-label='{esc(title)} calibration: forecast "
            "chance against how often it happened'>"
        )
    ]
    for v in (0, 0.25, 0.5, 0.75, 1):
        out.append(f"<line class='gridline' x1='{x(0)}' x2='{x(1)}' y1='{y(v):.1f}' y2='{y(v):.1f}'/>")
        out.append(f"<line class='gridline' y1='{y(0)}' y2='{y(1)}' x1='{x(v):.1f}' x2='{x(v):.1f}'/>")
    for v in (0, 0.5, 1):
        out.append(f"<text x='{x(v):.1f}' y='{size - 12}' text-anchor='middle'>{v:.0%}</text>")
        out.append(f"<text x='{pl - 6}' y='{y(v) + 3.5:.1f}' text-anchor='end'>{v:.0%}</text>")
    out.append(f"<line class='diag' x1='{x(0)}' y1='{y(0)}' x2='{x(1)}' y2='{y(1)}'/>")
    for r in rows:
        said, got = float(r["stated"]), float(r["observed"])
        lo, hi, n = float(r["ci_low"]), float(r["ci_high"]), int(r["n"])
        off = not (lo <= said <= hi)
        tip = f"Said {said:.0%}, happened {got:.0%} ({n:,} cases, 95% interval {lo:.0%} to {hi:.0%})"
        out.append(
            f"<g class='cpt{' off' if off else ''}'><title>{esc(tip)}</title>"
            f"<line class='ci' x1='{x(said):.1f}' x2='{x(said):.1f}' y1='{y(lo):.1f}' y2='{y(hi):.1f}'/>"
            f"<circle class='dot' cx='{x(said):.1f}' cy='{y(got):.1f}' r='4.5'/>"
            f"<circle class='hit' cx='{x(said):.1f}' cy='{y(got):.1f}' r='11'/></g>"
        )
    out.append(
        f"<text class='ax' x='{x(0.5):.1f}' y='{size - 1}' text-anchor='middle'>forecast chance</text>"
        f"<text class='ax' transform='translate(9 {y(0.5):.1f}) rotate(-90)' text-anchor='middle'>happened</text>"
        "</svg></figure>"
    )
    return "".join(out)


def calibration_plot(reliability: dict) -> str:
    """Win, podium and top-ten calibration as three small reliability diagrams."""
    panels = [
        _calibration_panel(label, reliability.get(k) or [])
        for k, label in (("win", "Win"), ("podium", "Podium"), ("top10", "Top 10"))
    ]
    panels = [p for p in panels if p]
    if not panels:
        return ""
    return (
        f"<div class='calgrid'>{''.join(panels)}</div>"
        "<div class='cal-key'><span><i class='k-diag'></i>perfect calibration</span>"
        "<span><i class='k-dot'></i>what happened, per band of forecasts</span>"
        "<span><i class='k-ci'></i>95% interval on how often it happened</span>"
        "<span><i class='k-off'></i>stated chance outside that interval</span></div>"
    )
