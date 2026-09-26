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

import pandas as pd

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
    """A probability as a percentage. Anything that rounds to zero is shown as
    below the smallest printable step rather than as 0, which reads as impossible."""
    floor = 10**-dp
    if 0 < x * 100 < floor / 2 or (x == 0):
        return f"&lt;{floor:g}%"
    # The mirror image: short of certain never prints as 100%.
    if x < 1 and x * 100 >= 100 - floor / 2:
        return f"&gt;{100 - floor:g}%"
    return f"{x * 100:.{dp}f}%"


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
  --paper:#faf9f6; --band:#f4f2ec; --chip:#eae7e0; --ink:#101012; --ink-2:#4e4e55; --ink-3:#67676d;
  --rule:#101012; --hair:#dcd9d3;
  --accent:#c8102e;            /* signal red: start lights and the live marker */
  --good:#12704a; --bad:#a8172e;
  --car-edge:rgba(16,16,18,.32);
  --grain:.030; --grain-blend:multiply;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:"Archivo","Helvetica Neue",Helvetica,Arial,sans-serif;
  __TEAM_LIGHT__
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0c0d11; --band:#121318; --chip:#1c1f26; --ink:#f2f2ef; --ink-2:#9fa1a8; --ink-3:#838389;
    --rule:#f2f2ef; --hair:#23262d;
    --accent:#ff3040;
    --good:#3cc98a; --bad:#ff6b7a;
    --car-edge:rgba(255,255,255,.26);
    --grain:.055; --grain-blend:overlay;
    __TEAM_DARK__
  }
}
:root[data-theme="dark"]{
  --paper:#0c0d11; --band:#121318; --chip:#1c1f26; --ink:#f2f2ef; --ink-2:#9fa1a8; --ink-3:#838389;
  --rule:#f2f2ef; --hair:#23262d;
  --accent:#ff3040;
  --good:#3cc98a; --bad:#ff6b7a;
  --car-edge:rgba(255,255,255,.26);
  --grain:.055; --grain-blend:overlay;
  __TEAM_DARK__
}

body{margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:15px; line-height:1.5;
  -webkit-font-smoothing:antialiased}

/* Faint noise on the dark theme so large dark areas don't look flat. */
body::before{content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
  opacity:var(--grain); mix-blend-mode:var(--grain-blend);
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)'/%3E%3C/svg%3E")}

.wrap{max-width:1140px; margin:0 auto; padding-inline:20px; padding-block:0 0}
h1,h2,h3{margin:0; text-wrap:balance; letter-spacing:-.02em; font-weight:700}
p{margin:0}
.mono{font-family:var(--mono); font-variant-numeric:tabular-nums}

/* ---- top bar: solid, dense, no blur. A broadcast strap, not a web header. */
.bar{background:var(--rule); color:var(--paper)}
.bar .inner{max-width:1140px; margin:0 auto; padding:0 20px; min-height:34px;
  display:flex; flex-wrap:wrap; align-items:center; gap:6px 20px;
  font-family:var(--mono); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase}
.bar .sep{flex:1}
.bar b{font-weight:600}
/* The bar inverts the page, so its link uses the bar's ink. */
.bar a{color:var(--paper); text-decoration:none; opacity:.85;
  border-bottom:1px solid color-mix(in srgb, var(--paper) 40%, transparent)}
.bar a:hover{opacity:1; border-bottom-color:var(--paper)}
.lights{display:inline-flex; gap:3px; align-items:center; margin-right:4px}
.lights i{width:6px; height:6px; border-radius:50%;
  background:color-mix(in srgb, var(--paper) 22%, transparent);
  animation:lights 3.2s ease-in-out 1 forwards; animation-delay:calc(var(--n) * 300ms)}
@keyframes lights{
  0%{background:color-mix(in srgb,var(--paper) 22%,transparent)}
  9%,45%{background:#ff2d20; box-shadow:0 0 6px #ff2d2099}
  53%,100%{background:color-mix(in srgb,var(--paper) 22%,transparent); box-shadow:none}
}

/* ---- masthead --------------------------------------------------------- */
.mast{padding:52px 0 34px; border-bottom:2px solid var(--rule)}
.mast h1{font-size:clamp(2.3rem,7vw,4.4rem); line-height:.96; font-weight:800;
  letter-spacing:-.035em}
.kicker{font-family:var(--mono); font-size:11px; letter-spacing:.2em;
  text-transform:uppercase; color:var(--ink-3); margin-bottom:14px}
.mast .sub{margin-top:16px; max-width:56ch; color:var(--ink-2); font-size:1rem}
.facts{display:flex; flex-wrap:wrap; gap:0 36px; margin-top:26px;
  padding-top:18px; border-top:2px solid var(--rule)}
.facts div{padding-block:4px}
.facts dt{font-family:var(--mono); font-size:9.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ink-3)}
.facts dd{margin:2px 0 0; font-family:var(--mono); font-size:.95rem; font-weight:500}

/* ---- section: label in the margin, content beside it ------------------- */
/* Alternate sections sit on a slightly darker band; the hairline does the separating. */
section{display:grid; grid-template-columns:132px 1fr; gap:0 28px;
  padding-block:44px; border-top:1px solid var(--hair); position:relative}
section::before{content:""; position:absolute; inset:0; z-index:-1;
  left:50%; transform:translateX(-50%); width:100vw; max-width:100vw;
  background:var(--band); opacity:0}
section.band::before{opacity:1}
section > .lab{font-family:var(--mono); font-size:10.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ink-3); padding-top:3px}
section > .lab b{display:block; color:var(--ink); font-weight:600; font-size:11px;
  letter-spacing:.12em; margin-bottom:6px}
section > .body{min-width:0}
.cap{color:var(--ink-2); font-size:.9rem; max-width:64ch; margin-bottom:18px}
/* Space between a table and a caption under it. */
.scroll + .cap{margin-top:22px}
.cap a, footer a{color:var(--ink); text-decoration:underline;
  text-decoration-color:var(--ink-3); text-underline-offset:3px;
  text-decoration-thickness:1px}
.cap a:hover, footer a:hover{text-decoration-color:var(--accent)}
@media (max-width:760px){
  section{grid-template-columns:1fr; gap:10px}
  section > .lab{padding-top:0}
}

/* ---- data tables: rules, not cards ------------------------------------ */
.grid5{display:grid;
  grid-template-columns:28px 3px minmax(110px,1.5fr) 52px 64px 60px 60px;
  align-items:center; gap:0 10px}
.grid4{display:grid; grid-template-columns:28px 3px minmax(110px,1.6fr) 60px 56px 72px;
  align-items:center; gap:0 10px}
.colhead{padding:0 0 7px; border-bottom:2px solid var(--rule);
  font-family:var(--mono); font-size:9.5px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--ink-3)}
.colhead span{text-align:right}
.colhead span:nth-child(3){text-align:left}
.row{padding:9px 0; border-bottom:1px solid var(--hair);
  transition:background .18s, box-shadow .18s;
  animation:slide .4s cubic-bezier(.2,.7,.3,1) backwards;
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
.team{font-family:var(--mono); font-size:9.5px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink-3)}
.v{font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right;
  font-size:.86rem}
.v.lead{font-weight:600; font-size:.95rem}
.v.dim{color:var(--ink-3)}
/* 10th-90th percentile range under the projected total. */
.rng{display:block; font-size:9.5px; font-weight:400; color:var(--ink-3);
  letter-spacing:.02em; margin-top:1px}

.gridC{display:grid; grid-template-columns:22px 3px minmax(90px,1fr) 52px 66px 52px;
  align-items:center; gap:0 10px}

@media (max-width:860px){
  .grid5{grid-template-columns:26px 3px minmax(96px,1fr) 52px 66px 56px}
  .grid5 > .c-points, .colhead.grid5 > span:nth-child(7){display:none}
}
@media (max-width:480px){
  .grid5{grid-template-columns:20px 3px minmax(70px,1fr) 34px 54px 42px; gap:0 8px}
  .grid4{grid-template-columns:22px 3px minmax(74px,1fr) 54px 66px; gap:0 8px}
  .grid4 > .c-third, .colhead.grid4 > span:nth-child(5){display:none}
  /* Too wide for a 320px screen with every column; drop "Now". */
  .gridC{grid-template-columns:20px 3px minmax(70px,1fr) 66px 50px; gap:0 8px}
  .gridC > .c-now, .colhead.gridC > span:nth-child(4){display:none}
  .wrap{padding-inline:14px}
  .bar .inner{padding:0 14px}
  /* The chart scrolls sideways rather than shrinking its labels to nothing. */
  .chart{overflow-x:auto}
  .chart svg{min-width:620px}
}

/* ---- the title arithmetic --------------------------------------------- */
.titlerace{display:flex; flex-wrap:wrap; align-items:baseline; gap:12px 34px;
  padding:14px 0 20px; margin-bottom:22px; border-bottom:1px solid var(--hair)}
.titlerace dt{font-family:var(--mono); font-size:9.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ink-3)}
.titlerace dd{margin:3px 0 0; font-family:var(--mono); font-size:1.45rem;
  font-weight:600; letter-spacing:-.02em; font-variant-numeric:tabular-nums}
.titlerace dd .of{font-size:.72rem; font-weight:400; color:var(--ink-3);
  margin-left:5px; letter-spacing:.04em}
.titlerace .verdict{flex:1 1 240px; color:var(--ink-2); font-size:.92rem;
  align-self:center}
.titlerace .verdict b{color:var(--ink); font-weight:600}

/* ---- two panels side by side (championship) --------------------------- */
.pair{display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:34px}
.panel h3{font-family:var(--mono); font-size:10.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ink-2); margin-bottom:10px}

/* ---- track record strip ------------------------------------------------ */
.strip{margin-bottom:18px}
.strip svg{display:block; max-width:100%; height:auto; overflow:visible}
.strip .rec rect{transition:opacity .15s}
.strip .rec.hit rect{fill:var(--good)}
.strip .rec.miss rect{fill:none; stroke:var(--ink-3); stroke-width:1}
.strip .rec:hover rect{opacity:.65}
.strip svg line{stroke:var(--hair); stroke-width:1}
.strip .key{margin-top:8px; font-family:var(--mono); font-size:10px; color:var(--ink-3)}
.strip .sw{display:inline-block; width:9px; height:9px; border-radius:2px; margin:0 5px 0 0;
  vertical-align:-1px}
.strip .sw.hit{background:var(--good)}
.strip .sw.miss{border:1px solid var(--ink-3); margin-left:14px}

/* ---- plain tables ------------------------------------------------------ */
/* 10px gutter for the emphasis marker, outside the text margin. */
.scroll{overflow-x:auto; padding-left:10px; margin-left:-10px}
table{border-collapse:collapse; width:100%; font-size:.86rem}
/* Gaps between columns, not after them, so both edges sit on the margins. */
th,td{padding:8px 0; text-align:left; border-bottom:1px solid var(--hair);
  white-space:nowrap; vertical-align:baseline}
th+th,td+td{padding-left:26px}
th:first-child,td:first-child{padding-left:0}
/* Figure tables: the label column takes the slack so figures group on the right.
   Key-value tables (label + paragraph) are sized the other way. */
.fig th:first-child,.fig td:first-child{width:99%}
.kv td:first-child{width:170px; vertical-align:top; padding-top:9px}
.kv td.feat{width:auto}
thead th{font-family:var(--mono); font-size:9.5px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--ink-3); font-weight:600;
  border-bottom:2px solid var(--rule)}
tbody tr:last-child td{border-bottom:0}
td.n,th.n{text-align:right}
td.n{font-family:var(--mono); font-variant-numeric:tabular-nums}
tr.me td{font-weight:600}
tr.me td:first-child{box-shadow:-10px 0 0 -7px var(--ink)}
.best{color:var(--good)}

/* ---- chart ------------------------------------------------------------- */
.chart{position:relative; width:100%}
.chart svg{display:block; width:100%; height:auto; overflow:visible}
.chart .gridline{stroke:var(--hair); stroke-width:1}
.chart .band{stroke:none; opacity:.13}
.chart .axis{stroke:var(--hair); stroke-width:1}
.chart text{font-family:var(--mono); font-size:10px; fill:var(--ink-3)}
.chart text.lbl{font-size:10.5px; font-weight:600; fill:var(--ink-2)}
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
.tip .rd{color:var(--ink-3); letter-spacing:.1em; text-transform:uppercase;
  font-size:9.5px; margin-bottom:5px}
.tip .r{display:flex; align-items:center; gap:7px; justify-content:space-between}
.tip .r i{width:8px; height:2px; border-radius:1px; flex:none}
.tip .r b{font-weight:600; margin-left:auto; font-variant-numeric:tabular-nums}
.tip .r em{font-style:normal; color:var(--ink-3); font-size:9.5px}
.legend{display:flex; flex-wrap:wrap; gap:6px 16px; margin-top:14px;
  font-family:var(--mono); font-size:10.5px; color:var(--ink-2)}
.legend span{display:inline-flex; align-items:center; gap:6px}
.legend i{width:14px; height:3px; border-radius:1px}

/* ---- numbered pipeline: these ARE a sequence, which is why they are numbered */
.steps{list-style:none; margin:0; padding:0; display:grid; gap:22px}
.steps li{display:grid; grid-template-columns:20px 1fr; gap:0 16px; align-items:start}
.steps .sn{font-family:var(--mono); font-size:12px; font-weight:600; color:var(--ink-3);
  font-variant-numeric:tabular-nums; text-align:right; line-height:1.5}
.steps h3{font-size:.98rem; margin-bottom:5px}
.steps p{color:var(--ink-2); font-size:.92rem; max-width:70ch}
.steps code, .feat code, .cap code{font-family:var(--mono); font-size:.82em;
  background:var(--chip); padding:1px 5px; border-radius:2px; color:var(--ink)}
td.feat{white-space:normal; color:var(--ink-2); line-height:1.7}
.lab span{display:block; margin-top:4px}

/* ---- footer ------------------------------------------------------------ */
.notes{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:20px 34px}
.notes p{color:var(--ink-2); font-size:.86rem}
.notes b{color:var(--ink); font-weight:600}
footer{border-top:2px solid var(--rule); margin-top:8px; padding:20px 0 56px;
  display:flex; flex-wrap:wrap; gap:8px 24px; justify-content:space-between;
  font-family:var(--mono); font-size:10px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-3)}
footer a{color:var(--ink-2)}

/* ---- status chips under the title ---------------------------------------- */
.status{display:flex; flex-wrap:wrap; gap:8px; margin-top:20px}
.chip{display:inline-flex; align-items:center; gap:7px; padding:4px 10px 4px 9px;
  border:1px solid var(--hair); border-radius:999px; background:var(--paper);
  font-family:var(--mono); font-size:10.5px; letter-spacing:.06em; color:var(--ink-2)}
.chip i{width:7px; height:7px; border-radius:50%; background:var(--ink-3); flex:none}
.chip.ok i{background:var(--good)}
.chip.warn i{background:#c98a00}
.chip.live i{background:var(--accent)}
.chip b{color:var(--ink); font-weight:600}

/* ---- headline facts ------------------------------------------------------ */
.headline{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:18px 30px;
  margin-top:26px; padding-top:18px; border-top:1px solid var(--hair)}
.headline dt{font-family:var(--mono); font-size:9.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ink-3)}
.headline dd{margin:4px 0 0; font-size:1.02rem; font-weight:600; line-height:1.3}
.headline dd .big{font-family:var(--mono); font-size:1.6rem; font-weight:600;
  letter-spacing:-.02em; margin-right:8px; font-variant-numeric:tabular-nums}
.headline dd small{display:block; font-weight:400; color:var(--ink-3); font-size:.8rem; margin-top:2px}

/* ---- the race board: every column read off one distribution -------------- */
.grid6{display:grid;
  grid-template-columns:28px 3px minmax(120px,1.6fr) 58px 104px 56px 56px 56px 54px 16px;
  align-items:center; gap:0 10px}
.board details{border-bottom:1px solid var(--hair)}
.board details:last-child{border-bottom:0}
.board summary{list-style:none; cursor:pointer; border-bottom:0}
.board summary::-webkit-details-marker{display:none}
.board summary:focus-visible{outline:2px solid var(--ink); outline-offset:2px}
.board .row{border-bottom:0}
.twist{font-family:var(--mono); font-size:12px; color:var(--ink-3); text-align:center;
  transition:transform .18s}
.board details[open] .twist{transform:rotate(90deg); color:var(--ink)}
.start small{display:block; font-size:9px; color:var(--ink-3); letter-spacing:.02em}
.winc{display:flex; flex-direction:column; align-items:flex-end; gap:4px}
.pbar{display:block; width:100%; height:3px; border-radius:2px; background:var(--chip); overflow:hidden}
.pbar i{display:block; height:100%; border-radius:2px; background:var(--tc,var(--ink));
  transform-origin:left; animation:grow .7s cubic-bezier(.2,.7,.3,1) backwards;
  animation-delay:calc(var(--i,0) * 35ms + 120ms)}
@keyframes grow{from{transform:scaleX(0)}}
.why{padding:4px 0 16px 41px; display:grid; grid-template-columns:minmax(0,1fr); gap:10px}
.why p{color:var(--ink-2); font-size:.88rem}
.contrib{display:grid; grid-template-columns:minmax(120px,190px) 1fr 52px; gap:6px 12px;
  align-items:center; max-width:560px; font-size:.8rem}
.contrib span{color:var(--ink-2)}
.contrib .axis{position:relative; height:10px; border-left:0}
.contrib .axis::before{content:""; position:absolute; left:50%; top:-3px; bottom:-3px;
  width:1px; background:var(--hair)}
.contrib .axis i{position:absolute; top:1px; height:8px; border-radius:2px}
.contrib .axis i.up{left:50%; background:var(--good)}
.contrib .axis i.down{right:50%; background:var(--bad)}
.contrib em{font-family:var(--mono); font-style:normal; font-size:.76rem; text-align:right;
  color:var(--ink-3); font-variant-numeric:tabular-nums}
.why .note{font-size:.76rem; color:var(--ink-3)}
@media (max-width:860px){
  .grid6{grid-template-columns:26px 3px minmax(96px,1fr) 50px 92px 52px 50px 14px}
  .grid6 > .c-top5, .grid6 > .c-top10{display:none}
}
@media (max-width:480px){
  .grid6{grid-template-columns:18px 3px minmax(64px,1fr) 38px 70px 44px 40px 10px; gap:0 7px}
  .why{padding-left:0}
  .contrib{grid-template-columns:minmax(90px,1fr) 1fr 42px}
}

/* ---- empty and notice states --------------------------------------------- */
.notice{border:1px solid var(--hair); border-left:3px solid var(--ink-3); padding:14px 16px;
  color:var(--ink-2); font-size:.9rem; max-width:64ch; background:var(--paper)}
.notice b{color:var(--ink)}

/* ---- method page: figures -------------------------------------------------- */
.figs{display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:26px 34px; margin:6px 0 10px}
.fig-rel svg{display:block; width:100%; height:auto; max-width:420px}
.fig-rel h4{font-family:var(--mono); font-size:10.5px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--ink-2); margin:0 0 8px; font-weight:600}
.fig-rel .diag{stroke:var(--ink-3); stroke-dasharray:3 4; stroke-width:1}
.fig-rel .frame{stroke:var(--hair); fill:none}
.fig-rel .ci{stroke:var(--ink-2); stroke-width:1.2}
.fig-rel .dot{fill:var(--accent); stroke:var(--paper); stroke-width:1.5}
.fig-rel text{font-family:var(--mono); font-size:9.5px; fill:var(--ink-3)}
.verdicts{display:flex; flex-wrap:wrap; gap:8px; margin:0 0 18px}
.pill{font-family:var(--mono); font-size:10.5px; padding:3px 9px; border-radius:999px;
  border:1px solid var(--hair); color:var(--ink-2)}
.pill.model{border-color:color-mix(in srgb,var(--good) 55%,var(--hair)); color:var(--good)}
.pill.grid{border-color:color-mix(in srgb,var(--bad) 55%,var(--hair)); color:var(--bad)}

@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important; transition:none!important}
}
"""

CSS = CSS.replace("__TEAM_LIGHT__", team_tokens(0)).replace("__TEAM_DARK__", team_tokens(1))

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Archivo:wght@400;500;600;700;800&"
    'family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
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


def contributions(detail: dict | None, scale: float) -> str:
    """Grouped SHAP contributions as diverging bars: right pushes the driver up
    the order, left down. Values are in the same units as the ranking score."""
    if not detail:
        return ""
    rows = []
    for label, value in detail.items():
        width = min(abs(float(value)) / scale, 1.0) * 50 if scale > 0 else 0.0
        cls = "up" if value > 0 else "down"
        rows.append(
            f"<span>{esc(label)}</span>"
            f"<div class='axis'><i class='{cls}' style='width:{width:.1f}%'></i></div>"
            f"<em>{float(value):+.2f}</em>"
        )
    return "<div class='contrib'>" + "".join(rows) + "</div>"


def race_board(rows: list[dict], details: dict[str, dict] | None = None) -> str:
    """The race forecast. Every figure in a row is read off the same finishing
    distribution, so win <= podium <= top 5 <= top 10 always holds.

    Each row opens to say why: the SHAP contributions behind the driver's
    score, grouped so near-duplicate features don't split the credit.
    """
    details = details or {}
    scale = max(
        (abs(float(v)) for d in details.values() for v in (d or {}).values()),
        default=0.0,
    )
    out = [
        "<div class='board'>",
        (
            "<div class='colhead grid6'><span></span><span></span><span>Driver</span>"
            "<span>Start</span><span>Win</span><span>Podium</span><span class='c-top5'>Top 5</span>"
            "<span class='c-top10'>Top 10</span><span>Exp.</span><span></span></div>"
        ),
    ]
    for i, r in enumerate(rows):
        p = float(r.get("p_win") or 0)
        exp = r.get("exp_position")
        why = r.get("why") or ""
        detail = details.get(r.get("driver_id"))
        out.append(
            f"<details><summary class='row grid6{' podium' if i < 3 else ''}' "
            f'style="--i:{i}; --tc:{team_colour(r.get("team"))}">'
            + _row_head(i, r)
            + _start_cell(r)
            + f"<div class='v lead winc'><span data-count>{pct(p)}</span>"
            + f"<span class='pbar'><i style='width:{min(p, 1.0) * 100:.1f}%'></i></span></div>"
            + f"<div class='v'>{pct(r.get('p_podium') or 0, 0)}</div>"
            + f"<div class='v dim c-top5'>{pct(r.get('p_top5') or 0, 0)}</div>"
            + f"<div class='v dim c-top10'>{pct(r.get('p_top10') or 0, 0)}</div>"
            + (
                f"<div class='v dim'>{float(exp):.1f}</div>"
                if isinstance(exp, (int, float))
                else "<div class='v dim'>&mdash;</div>"
            )
            + "<div class='twist' aria-hidden='true'>&rsaquo;</div>"
            + "</summary>"
            + "<div class='why'>"
            + (f"<p>{esc(why)}.</p>" if why else "<p>No explanation was logged with this forecast.</p>")
            + contributions(detail, scale)
            + (
                "<p class='note'>What the model leaned on for this forecast (SHAP, relative to the "
                "field average) &mdash; not what causes a result.</p>"
                if detail
                else ""
            )
            + "</div></details>"
        )
    out.append("</div>")
    return "".join(out)


def quali_board(rows: list[dict], qualified: bool = False) -> str:
    """The qualifying forecast. Once qualifying has run, the last column shows
    where each driver actually qualified instead of their top-ten chance."""
    last = "Qualified" if qualified else "Top 10"
    out = [
        (
            "<div class='colhead grid4'><span></span><span></span><span>Driver</span>"
            f"<span>Pole</span><span>Top 3</span><span>{last}</span></div>"
        )
    ]
    for i, r in enumerate(rows):
        if qualified:
            g = r.get("grid")
            end = (
                f"<div class='v'>P{int(g)}</div>"
                if isinstance(g, (int, float))
                else "<div class='v dim'>&mdash;</div>"
            )
        else:
            end = f"<div class='v dim'>{pct(r.get('p_top10') or 0, 0)}</div>"
        out.append(
            f"<div class='row grid4{' podium' if i < 3 else ''}' "
            f'style="--i:{i}; --tc:{team_colour(r.get("team"))}">'
            + _row_head(i, r)
            + f"<div class='v lead' data-count>{pct(r.get('p_win') or 0)}</div>"
            + f"<div class='v c-third'>{pct(r.get('p_podium') or 0, 0)}</div>"
            + end
            + "</div>"
        )
    return "".join(out)


def _odds(p: float, alive: bool = True) -> str:
    """Title odds, held between <0.1% and 99.9%.

    10,000 simulated seasons can't resolve 1 in 10,000, so a driver who can
    still win it mathematically is never shown at 0 or 100. allocate_odds only
    hands out 100% once the title is settled.
    """
    if not alive:
        return "out"
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
            + f"<div class='v'>{_odds(r['p_title'], r.get('alive', True))}</div>"
            + "</div>"
        )
    return "".join(out) + "</div>"


def title_race(outlook: dict, leader: str, rival: str) -> str:
    """The arithmetic of the title, which is what people actually argue about.

    Not a model output - pure subtraction against the points still on the
    table. It is the one number on this page that cannot be wrong.
    """
    lead = outlook.get("lead") or 0
    avail = outlook.get("points_available") or 0
    k = outlook.get("clinch_in")
    after = outlook.get("after_round") or 0

    races_left = outlook.get("n_races") or 0
    if outlook.get("clinched"):
        verdict = f"<b>{esc(leader)} has already clinched it.</b>"
    elif k and k >= races_left:
        verdict = "On the arithmetic alone, this one goes to the <b>final race</b>."
    elif k:
        verdict = (
            f"Earliest {esc(leader)} can mathematically seal it: "
            f"<b>round {after + k}</b>, {k} race{'s' if k > 1 else ''} from now."
        )
    else:
        verdict = "The season is over."

    wins = outlook.get("leader_wins")
    win_cell = (
        f"<div><dt>{esc(leader)}, mean wins</dt>"
        f"<dd>{wins:.1f}<span class='of'>of {races_left}</span></dd></div>"
        if wins is not None
        else ""
    )
    return (
        "<div class='titlerace'>"
        f"<div><dt>Lead over {esc(rival)}</dt><dd>{lead:.0f}<span class='of'>pts</span></dd></div>"
        f"<div><dt>Still available</dt><dd>{avail}<span class='of'>pts</span></dd></div>"
        + win_cell
        + f"<div class='verdict'>{verdict}</div>"
        "</div>"
    )


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


def record_strip(board) -> str:
    """One bar per graded race: height is the probability published for the
    model's pick, fill is whether it won.
    """
    if board is None or board.empty or "winner_hit" not in board:
        return ""
    rows = list(board.itertuples())
    w, h, gap = 10.0, 44.0, 4.0
    width = len(rows) * (w + gap)
    bars = []
    for i, r in enumerate(rows):
        p = float(getattr(r, "p_top_pick", 0.0) or 0.0)
        p = min(max(p, 0.0), 1.0)
        bh = max(3.0, p * h)
        x = i * (w + gap)
        hit = bool(getattr(r, "winner_hit", 0))
        cls = "rec hit" if hit else "rec miss"
        label = f"{getattr(r, 'race', '')}: said {p:.0%} for {getattr(r, 'picked', '?')}, "
        label += "correct" if hit else f"{getattr(r, 'actual', '?')} won"
        bars.append(
            f"<g class='{cls}'><title>{esc(label)}</title>"
            f"<rect x='{x:.1f}' y='{h - bh:.1f}' width='{w}' height='{bh:.1f}' rx='1.5'/></g>"
        )
    # Drawn at its own pixel size: scaled to the column, four races became four
    # giant pills and a band of empty page.
    return (
        f"<div class='strip'><svg viewBox='0 0 {max(width, 1):.0f} {h + 12:.0f}' "
        f"width='{max(width, 1):.0f}' height='{h + 12:.0f}' "
        f"preserveAspectRatio='xMinYMid meet' role='img' "
        f"aria-label='Winner called or missed, per graded race'>"
        + "".join(bars)
        + f"<line x1='0' y1='{h + 0.5}' x2='{max(width, 1):.0f}' y2='{h + 0.5}'/></svg>"
        "<p class='key'><span class='sw hit'></span>winner called"
        "<span class='sw miss'></span>missed &middot; bar height is the probability published</p></div>"
    )


def _is_numeric_column(series: pd.Series) -> bool:
    """Whether a column holds quantities, which decides its alignment.

    Strings count when a unit is all that stands between them and a number,
    e.g. "71%" or "90 pts".
    """
    if series.empty or pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    stripped = series.astype(str).str.replace(r"[,%]|\s*(pts|pt|races|race)\s*$", "", regex=True)
    return bool(pd.to_numeric(stripped.str.strip(), errors="coerce").notna().all())


def _decimals(series: pd.Series) -> int:
    """Decimal places for a whole column, from the value that needs the most, so
    0.890 doesn't print as 0.89 beside 0.881.
    """
    numbers = [v for v in series if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numbers:
        return 0
    return max(len(f"{float(v):.6f}".rstrip("0").partition(".")[2]) for v in numbers)


def table(df: pd.DataFrame, emphasise: str | None = None, best_cols: dict | None = None) -> str:
    """best_cols maps column name -> 'min' or 'max' to mark the leading value."""
    if df is None or df.empty:
        return ""
    best_cols = best_cols or {}
    winners = {}
    for col, how in best_cols.items():
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                winners[col] = series.min() if how == "min" else series.max()

    # First column is the row label, always left. Other columns align on their
    # content, and each heading takes its column's alignment.
    align = [False] + [_is_numeric_column(df[c]) for c in df.columns[1:]]
    # Figures loaded from JSON arrive as strings and each kept its own decimal
    # count, so "0.32" sat in a column of "0.164" and "0.481". A column that
    # parses cleanly as numbers - no unit suffix to preserve - becomes numbers.
    df = df.copy()
    for c, right in zip(df.columns, align):
        if right and not pd.api.types.is_numeric_dtype(df[c]):
            parsed = pd.to_numeric(df[c], errors="coerce")
            if parsed.notna().all():
                df[c] = parsed
    places = {c: _decimals(df[c]) for c in df.columns}

    head = "".join(
        f"<th class='n'>{esc(c)}</th>" if right else f"<th>{esc(c)}</th>"
        for c, right in zip(df.columns, align)
    )
    body = []
    for _, r in df.iterrows():
        cls = " class='me'" if emphasise and str(r.iloc[0]) == emphasise else ""
        cells = []
        for j, c in enumerate(df.columns):
            v = r[c]
            numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
            mark = " best" if c in winners and numeric and abs(v - winners[c]) < 1e-9 else ""
            text = f"{v:.{places[c]}f}" if numeric else esc(v)
            cells.append(f"<td class='{'n' if align[j] else ''}{mark}'>{text}</td>")
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f"<div class='scroll'><table class='fig'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def top_bar(prediction: dict | None, generated: datetime) -> str:
    """A broadcast strap: solid, dense, one line of state."""
    bits = ["<span class='lights'>" + "".join(f"<i style='--n:{n}'></i>" for n in range(5)) + "</span>"]
    if prediction:
        bits.append(f"<b>{'Grid known' if prediction.get('grid_known') else 'Before qualifying'}</b>")
        start = prediction.get("race_start_utc")
        if start:
            bits.append(f"<span data-countdown='{esc(start)}'>&mdash;</span>")
    bits.append("<span class='sep'></span>")
    bits.append(f"<span data-since='{generated.isoformat()}'>updated {generated:%d %b %H:%M} UTC</span>")
    return f"<div class='bar'><div class='inner'>{''.join(bits)}</div></div>"


# The page's only script. Three jobs, each of which is something static HTML
# genuinely cannot do: say how long ago the page was built, count down to the
# race, and read values off the chart.
SCRIPT = r"""
(function(){
  var rtf = null;
  try { rtf = new Intl.RelativeTimeFormat(undefined,{numeric:'auto'}); } catch(e){}

  function ago(el){
    var t = Date.parse(el.dataset.since); if(isNaN(t) || !rtf) return;
    var mins = Math.round((t - Date.now())/60000), txt;
    if(Math.abs(mins) < 60) txt = rtf.format(mins,'minute');
    else if(Math.abs(mins) < 2880) txt = rtf.format(Math.round(mins/60),'hour');
    else txt = rtf.format(Math.round(mins/1440),'day');
    el.textContent = 'updated ' + txt;
  }
  function counts(el){
    var t = Date.parse(el.dataset.countdown); if(isNaN(t)) return;
    var ms = t - Date.now();
    if(ms <= 0){ el.textContent = 'race under way'; return; }
    var d = Math.floor(ms/86400000), h = Math.floor(ms%86400000/3600000), m = Math.floor(ms%3600000/60000);
    el.textContent = 'lights out in ' + (d ? d+'d ' : '') + h + 'h ' + m + 'm';
  }
  function tick(){
    document.querySelectorAll('[data-since]').forEach(ago);
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


def document(body: str, standalone: bool = True, title: str = "Pit Wall") -> str:
    """standalone=True writes a complete file for GitHub Pages; False returns
    the fragment an artifact host wraps in its own skeleton."""
    if not standalone:
        return f"<title>{esc(title)}</title>{FONTS}<style>{CSS}</style>{body}<script>{SCRIPT}</script>"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1,viewport-fit=cover'>"
        f"<title>{esc(title)}</title>{FONTS}<style>{CSS}</style></head>"
        f"<body>{body}<script>{SCRIPT}</script></body></html>"
    )


def json_payload(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def utcnow() -> datetime:
    return datetime.now(UTC)


def reliability_chart(rows: list[dict], title: str, size: int = 260) -> str:
    """Stated probability against how often it happened, one dot per bucket,
    with its 95% interval. On the diagonal is calibrated; below it, over-confident.

    Axes are square-root scaled so the long tail of small probabilities -
    where most of a 22-car field lives - isn't crushed into a corner.
    """
    if not rows:
        return ""
    pad, plot = 30, size - 40

    def sc(v: float) -> float:
        return float(max(v, 0.0)) ** 0.5

    def X(v: float) -> float:
        return pad + sc(v) * plot

    def Y(v: float) -> float:
        return 10 + plot - sc(v) * plot

    parts = [
        f"<rect class='frame' x='{pad}' y='10' width='{plot}' height='{plot}'/>",
        f"<line class='diag' x1='{X(0)}' y1='{Y(0)}' x2='{X(1)}' y2='{Y(1)}'/>",
    ]
    for t in (0.05, 0.25, 0.5, 1.0):
        parts.append(f"<text x='{X(t):.1f}' y='{size - 14}' text-anchor='middle'>{t:.0%}</text>")
        parts.append(f"<text x='{pad - 4}' y='{Y(t) + 3:.1f}' text-anchor='end'>{t:.0%}</text>")
    for r in rows:
        stated, observed = float(r["stated"]), float(r["observed"])
        lo, hi, n = float(r["ci_low"]), float(r["ci_high"]), int(r["n"])
        parts.append(
            f"<g><title>stated {stated:.1%}, happened {observed:.1%} (n={n}, 95% interval "
            f"{lo:.0%}-{hi:.0%})</title>"
            f"<line class='ci' x1='{X(stated):.1f}' x2='{X(stated):.1f}' y1='{Y(lo):.1f}' y2='{Y(hi):.1f}'/>"
            f"<circle class='dot' cx='{X(stated):.1f}' cy='{Y(observed):.1f}' r='{2.5 + min(n, 400) ** 0.5 / 4:.1f}'/></g>"
        )
    parts.append(f"<text x='{pad + plot / 2}' y='{size - 1}' text-anchor='middle'>stated</text>")
    return (
        f"<figure class='fig-rel'><h4>{esc(title)}</h4>"
        f"<svg viewBox='0 0 {size} {size + 4}' role='img' aria-label='{esc(title)} reliability'>"
        + "".join(parts)
        + "</svg></figure>"
    )
