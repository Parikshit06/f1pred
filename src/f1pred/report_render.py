"""Presentation layer for the dashboard.

Visual direction is the pit wall, not the broadsheet. Every convention here is
borrowed from a live timing screen because that is the native document of this
subject: fixed-width numerals that hold their column as they tick, constructor
colour used as the only chroma so a car is identifiable before you read the
name, and gaps drawn as bars because a gap is a length.

Two rules follow from that and are worth stating, since both are easy to
violate later:

  * Colour is data. Constructor colour identifies a car; the accent marks the
    model's own voice; green and red mean better and worse than a baseline.
    Nothing is coloured for decoration, which is what keeps twenty-two team
    colours from turning into noise.
  * Numbers are monospaced and right-aligned, always. A probability column you
    cannot scan vertically is a probability column nobody reads.

Self-contained: inline CSS, hand-built SVG, no CDN, no build step. Renders
identically from GitHub Pages, from a file:// URL, and offline.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime

import pandas as pd

# Constructor colours, one pair per team: (light surface, dark surface).
#
# A livery is not a palette. Mercedes petrol is 1.8:1 against white and Renault
# yellow is 1.15:1 - as a 2px line or a thin bar they are invisible, which is a
# correctness problem, not a taste one, because colour is the only thing
# identifying a car before you read the name. Each team therefore gets a variant
# per surface, shifted in lightness and kept on its own hue.
#
# The four that carry the chart - Mercedes, Ferrari, McLaren, Red Bull - were
# picked by running the pairs through a CVD validator rather than by eye. Both
# sets clear the lightness band, the chroma floor, the normal-vision separation
# floor and 3:1 against their surface. Ferrari red against McLaren orange sits
# in the 6-8 deuteranopia band, which is legal only with a second channel, so
# every line on the chart is also labelled at its end and teammates are split by
# dash pattern - the broadcast convention anyway.
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


def esc(x) -> str:
    return html.escape(str(x))


def pct(x: float, dp: int = 1) -> str:
    return f"{x * 100:.{dp}f}%"


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# Light is the FIA classification sheet: white stock, black figures, colour
# only on the constructor rule. Dark is the same document on the pit wall.
# Both are the subject's own vernacular, so neither is an afterthought.
CSS = """
*,*::before,*::after{box-sizing:border-box}
/* SVG elements ignore the HTML hidden attribute without this. Leaving it out
   parks the hover crosshair and its dots at the viewBox origin, which renders
   as a stray line and a stray dot in the corner of the chart. */
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

/* Depth on the dark theme comes from grain, not a gradient.
   Flat near-black reads as unfinished; a gradient wash reads as generated. A
   fine fractal-noise overlay at 3-5% is the print answer: it is invisible as an
   effect and it stops large dark areas looking like dead pixels. Generated
   inline as an SVG filter, so the page stays a single file with no CDN. */
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
/* The bar inverts the page, so its link takes the bar's own ink. A transparent
   colour-mix was used here and computed to roughly the bar's own background at
   some zoom levels; an opaque token cannot do that. */
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
/* Alternating full-bleed bands give the page rhythm without wrapping anything
   in a card.
   The step between paper and band is deliberately about 2.4 in L* - roughly
   what a second ink pass costs on press. It was 4.2, which is enough to read
   as a grey box dropped on the page rather than the same sheet under slightly
   different light; every band edge already carries a hairline, so the tone
   does not have to do the work of separating anything. */
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
  grid-template-columns:28px 3px minmax(110px,1.5fr) 52px 60px 56px 56px 92px;
  align-items:center; gap:0 10px}
.grid4{display:grid; grid-template-columns:28px 3px minmax(110px,1.6fr) 60px 56px 56px;
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
/* Hovering a row lifts its constructor colour into the margin - the car you
   are looking at, marked, without moving anything. */
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
/* The 10th-90th percentile under the projected total. A single number reads as
   a promise; the range is the actual claim. */
.rng{display:block; font-size:9.5px; font-weight:400; color:var(--ink-3);
  letter-spacing:.02em; margin-top:1px}

@media (max-width:860px){
  .grid5{grid-template-columns:26px 3px minmax(96px,1fr) 52px 66px 56px}
  .grid5 > .c-points, .colhead.grid5 > span:nth-child(7){display:none}
}
@media (max-width:480px){
  .grid5{grid-template-columns:22px 3px minmax(74px,1fr) 54px 46px; gap:0 8px}
  .grid5 > .c-start, .colhead.grid5 > span:nth-child(4){display:none}
  .grid4{grid-template-columns:22px 3px minmax(74px,1fr) 54px 46px; gap:0 8px}
  .grid4 > .c-third, .colhead.grid4 > span:nth-child(5){display:none}
  /* The championship panels carry three numeric columns and overflow a 320px
     screen if they keep them all. "Now" is the one a reader already knows. */
  .gridC,.gridT{grid-template-columns:20px 3px minmax(70px,1fr) 58px 50px; gap:0 8px}
  .gridC > .c-now, .colhead.gridC > span:nth-child(4){display:none}
  .wrap{padding-inline:14px}
  .bar .inner{padding:0 14px}
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
.gridC{display:grid; grid-template-columns:22px 3px minmax(90px,1fr) 52px 58px 52px;
  align-items:center; gap:0 10px}
.gridT{display:grid; grid-template-columns:22px 3px minmax(90px,1fr) 52px 58px 52px;
  align-items:center; gap:0 10px}

/* ---- plain tables ------------------------------------------------------ */
.scroll{overflow-x:auto}
table{border-collapse:collapse; width:100%; font-size:.86rem}
th,td{padding:8px 12px 8px 0; text-align:left; border-bottom:1px solid var(--hair);
  white-space:nowrap}
thead th{font-family:var(--mono); font-size:9.5px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--ink-3); font-weight:600;
  border-bottom:2px solid var(--rule)}
tbody tr:last-child td{border-bottom:0}
td.n{font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right}
th.n{text-align:right}
tr.me td{font-weight:600}
tr.me td:first-child{box-shadow:inset 3px 0 0 var(--ink); padding-left:10px}
.best{color:var(--good)}

/* ---- chart ------------------------------------------------------------- */
.chart{position:relative; width:100%}
.chart svg{display:block; width:100%; height:auto; overflow:visible}
.chart .gridline{stroke:var(--hair); stroke-width:1}
.chart .band{stroke:none; opacity:.13}
.chart .axis{stroke:var(--hair); stroke-width:1}
.chart text{font-family:var(--mono); font-size:10px; fill:var(--ink-3)}
.chart text.lbl{font-size:10.5px; font-weight:600; fill:var(--ink-2)}
.chart .now{stroke:var(--accent); stroke-width:1; stroke-dasharray:2 3}
/* Lines draw themselves in once, left to right, the way a lap builds. The
   dash offset is set from the measured path length in script so the timing is
   right whatever the shape. */
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
.legend i{width:14px; height:2px; background:currentColor; border-radius:1px}

/* ---- numbered pipeline: these ARE a sequence, which is why they are numbered */
.steps{list-style:none; margin:0; padding:0; display:grid; gap:22px}
.steps li{display:grid; grid-template-columns:30px 1fr; gap:0 16px; align-items:start}
.steps .sn{font-family:var(--mono); font-size:11px; font-weight:600; color:var(--ink-3);
  border-top:2px solid var(--ink); padding-top:5px; margin-top:4px}
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


def race_board(rows: list[dict]) -> str:
    out = [
        (
            "<div class='colhead grid5'><span></span><span></span><span>Driver</span>"
            "<span>Start</span><span>Win</span><span>Podium</span><span>Points</span></div>"
        )
    ]
    for i, r in enumerate(rows):
        grid = r.get("grid")
        start = f"P{int(grid)}" if isinstance(grid, (int, float)) else "&mdash;"
        out.append(
            f"<div class='row grid5{' podium' if i < 3 else ''}' "
            f'style="--i:{i}; --tc:{team_colour(r.get("team"))}">'
            + _row_head(i, r)
            + f"<div class='v dim c-start'>{start}</div>"
            + f"<div class='v lead' data-count>{pct(r.get('p_win') or 0)}</div>"
            + f"<div class='v'>{pct(r.get('p_podium') or 0, 0)}</div>"
            + f"<div class='v dim c-points'>{pct(r.get('p_top10') or 0, 0)}</div>"
            + "</div>"
        )
    return "".join(out)


def quali_board(rows: list[dict]) -> str:
    out = [
        (
            "<div class='colhead grid4'><span></span><span></span><span>Driver</span>"
            "<span>Pole</span><span>Top 3</span><span>Top 10</span></div>"
        )
    ]
    for i, r in enumerate(rows):
        out.append(
            f"<div class='row grid4{' podium' if i < 3 else ''}' "
            f'style="--i:{i}; --tc:{team_colour(r.get("team"))}">'
            + _row_head(i, r)
            + f"<div class='v lead' data-count>{pct(r.get('p_win') or 0)}</div>"
            + f"<div class='v c-third'>{pct(r.get('p_podium') or 0, 0)}</div>"
            + f"<div class='v dim'>{pct(r.get('p_top10') or 0, 0)}</div>"
            + "</div>"
        )
    return "".join(out)


def _odds(p: float) -> str:
    """Title odds, clamped away from certainty.

    A 10,000-run simulation cannot resolve below 1 in 10,000, and the real
    uncertainty nine races out is the assumption that current form holds - not
    the sampling. So the column never prints 100%: it tops out at 99.9%.
    """
    return f"{min(max(p, 0.0), 0.999) * 100:.1f}%"


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
            + f"<div class='v'>{_odds(r['p_title'])}</div>"
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
    """Cumulative championship points by round: solid where it happened, dashed
    where it is projected.

    One y-axis, one unit, no second scale.

    Dash means projected and nothing else - an earlier version also used it to
    split teammates, which made a driver's completed season read as a forecast.
    Teammates share a constructor colour and are separated by stroke weight plus
    the label at the end of every line, so colour alone never carries identity.
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

    pad_l, pad_r, pad_t, pad_b = 44, 118, 16, 30
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

    # Labels are placed on a simple collision ladder so two drivers finishing
    # within a few points of each other do not print on top of one another.
    ends = sorted(((s["points"][max(s["points"])], s) for s in series), key=lambda t: -t[0])
    taken: list[float] = []
    for value, s in ends:
        colour = team_colour(s["team"])
        # Second car in a garage: same hue, lighter stroke.
        weight = "1.4" if s.get("second_car") else "2"
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
            out.append(f"<path class='ser' d='{path(actual)}' stroke='{colour}' stroke-width='{weight}'/>")
        if len(future) > 1:
            out.append(
                f"<path class='ser proj' d='{path(future)}' stroke='{colour}' stroke-width='{weight}'/>"
            )
        if actual:
            r, v = actual[-1]
            out.append(f"<circle class='pt' cx='{X(r):.1f}' cy='{Y(v):.1f}' r='3.5' fill='{colour}'/>")

        # Each label is two lines - name over value - so the ladder clears 25px,
        # not the 13 an earlier version used, which stacked them on top of one
        # another whenever three drivers finished within a few points.
        y = Y(value)
        while any(abs(y - t) < 25 for t in taken):
            y += 25
        taken.append(y)
        out.append(
            f"<text class='lbl' x='{pad_l + plot_w + 10}' y='{y + 3.5:.1f}' text-anchor='start'>"
            f"{esc(s['label'])}</text>"
            f"<text x='{pad_l + plot_w + 10}' y='{y + 15:.1f}' text-anchor='start'>{value:.0f}</text>"
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
        f"<span style='color:{team_colour(s['team'])}'><i></i></span>"
        f"<span style='margin-left:-10px'>{esc(s['label'])}</span>"
        for s in series
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
                "colour": team_colour(s["team"]),
                "points": {str(r): round(v, 1) for r, v in sorted(s["points"].items())},
            }
            for s in series
        ],
    }
    out.append(f"<script type='application/json' class='chart-data'>{json.dumps(payload)}</script>")
    out.append("</div>")
    return "".join(out)


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

    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cls = " class='me'" if emphasise and str(r.iloc[0]) == emphasise else ""
        cells = []
        for j, c in enumerate(df.columns):
            v = r[c]
            numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
            mark = " best" if c in winners and numeric and abs(v - winners[c]) < 1e-9 else ""
            text = f"{v:g}" if numeric else esc(v)
            cells.append(f"<td class='{'n' if j else ''}{mark}'>{text}</td>")
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


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
