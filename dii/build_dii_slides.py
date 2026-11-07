"""Append the DII section to the existing update deck, matching its style."""
import copy
from pptx import Presentation
from pptx.util import Emu, Pt, Inches
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.oxml.ns import qn

NAVY, TEAL, MINT, LIGHT, GREY, BORDER, WHITE = (RGBColor.from_string(c) for c in
    ['0B1F3A', '1B9AAA', '5FE0D0', 'EAF0F5', '5B6B7B', 'D5DDE5', 'FFFFFF'])
RED = RGBColor.from_string('C0392B')
GREEN = RGBColor.from_string('1E8449')
EYEBROW = 'DRIVER INTUITIVENESS INDEX (DII)'
L, W = Inches(0.5), Inches(9.0)

import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else 'orig.pptx'          # the original update deck
DST = sys.argv[2] if len(sys.argv) > 2 else 'ACC_BCC_RL_Update_Extended_DII.pptx'
prs = Presentation(SRC)
layout = prs.slide_layouts[0]


def tb(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, name=None):
    """paras: list of (text or [(text, opts)], para_opts)."""
    box = slide.shapes.add_textbox(x, y, w, h)
    if name: box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, (runs, po) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(runs, str):
            runs = [(runs, {})]
        for text, o in runs:
            r = p.add_run()
            r.text = text
            f = r.font
            f.size = Pt(o.get('size', po.get('size', 11)))
            f.bold = o.get('bold', po.get('bold', False))
            f.italic = o.get('italic', po.get('italic', False))
            f.color.rgb = o.get('color', po.get('color', NAVY))
            if o.get('font', po.get('font')):
                f.name = o.get('font', po.get('font'))
        p.alignment = po.get('align', PP_ALIGN.LEFT)
        if po.get('after') is not None:
            p.space_after = Pt(po['after'])
        if po.get('bullet'):
            pPr = p._p.get_or_add_pPr()
            pPr.set('marL', '228600'); pPr.set('indent', '-228600')
            for tag in ('a:buSzPct', 'a:buChar'):
                pass
            bs = pPr.makeelement(qn('a:buSzPct'), {'val': '100000'}); pPr.append(bs)
            bc = pPr.makeelement(qn('a:buChar'), {'char': '•'}); pPr.append(bc)
    return box


def card(slide, x, y, w, h, fill=LIGHT, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.adjustments[0] = 0.04
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    if name: s.name = name
    return s


def new_slide(title, eyebrow=EYEBROW, notes=None):
    s = prs.slides.add_slide(layout)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = WHITE
    tb(s, L, Inches(0.25), W, Inches(0.25), [(eyebrow, {'size': 10, 'bold': True, 'color': TEAL})], MSO_ANCHOR.MIDDLE, 'Eyebrow')
    tb(s, L, Inches(0.5), W, Inches(0.6), [(title, {'size': 22, 'bold': True, 'font': 'Georgia'})], MSO_ANCHOR.MIDDLE, 'Title')
    if notes: s.notes_slide.notes_text_frame.text = notes
    return s


def footnote(s, text, y=Inches(5.05)):
    tb(s, L, y, W, Inches(0.35), [(text, {'size': 10.5, 'italic': True, 'color': GREY})], name='Source')


def table(s, x, y, w, rows, col_w=None, size=10, bold_cells=(), fills=None, row_h=Inches(0.27)):
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, x, y, w, row_h * nr)
    t = gt.table
    tblPr = t._tbl.tblPr
    for a in ('firstRow', 'bandRow'): tblPr.set(a, '0')
    st = tblPr.find(qn('a:tableStyleId'))
    if st is not None: tblPr.remove(st)
    if col_w:
        for i, cw in enumerate(col_w): t.columns[i].width = cw
    for r in range(nr):
        t.rows[r].height = row_h
        for c in range(nc):
            cell = t.cell(r, c)
            cell.margin_left = cell.margin_right = Emu(91440)
            cell.margin_top = cell.margin_bottom = Emu(36000)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame; tf.word_wrap = True
            tf.paragraphs[0].text = ''
            run = tf.paragraphs[0].add_run(); run.text = str(rows[r][c])
            run.font.size = Pt(size)
            hdr = r == 0
            run.font.bold = hdr or (r, c) in bold_cells
            run.font.color.rgb = WHITE if hdr else NAVY
            cell.fill.solid()
            f = NAVY if hdr else (LIGHT if r % 2 == 0 else WHITE)
            if fills and (r, c) in fills: f = fills[(r, c)]
            cell.fill.fore_color.rgb = f
            tcPr = cell._tc.get_or_add_tcPr()
            for ln in ('a:lnL', 'a:lnR', 'a:lnT', 'a:lnB'):
                e = tcPr.makeelement(qn(ln), {'w': '6350'})
                sf = e.makeelement(qn('a:solidFill'), {}); clr = sf.makeelement(qn('a:srgbClr'), {'val': 'D5DDE5'})
                sf.append(clr); e.append(sf); tcPr.insert(0, e) if False else tcPr.append(e)
            # schema order: borders must precede fill
            fill_el = tcPr.find(qn('a:solidFill'))
            if fill_el is not None:
                tcPr.remove(fill_el); tcPr.append(fill_el)
    return gt


def style_chart(ch, size=10):
    ch.font.size = Pt(size); ch.font.color.rgb = NAVY; ch.font.name = 'Calibri'


from pptx.enum.chart import XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE_DASH_STYLE


def bar_chart(s, x, y, w, h, cats, vals, title, colors, fmt, vmax, name):
    cd = CategoryChartData(); cd.categories = cats; cd.add_series('v', vals)
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, x, y, w, h, cd); gf.name = name
    c = gf.chart; style_chart(c); c.has_legend = False
    c.has_title = True; c.chart_title.text_frame.text = title
    r = c.chart_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(11); r.font.bold = True
    ca = c.category_axis; ca.reverse_order = True; ca.format.line.color.rgb = BORDER; ca.tick_labels.font.size = Pt(10)
    va = c.value_axis; va.visible = False; va.has_major_gridlines = False; va.minimum_scale = 0; va.maximum_scale = vmax
    pl = c.plots[0]; pl.gap_width = 55; pl.has_data_labels = True
    pl.data_labels.number_format = fmt; pl.data_labels.number_format_is_linked = False
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END; pl.data_labels.font.size = Pt(10)
    for i, col in enumerate(colors):
        pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = col
    return c


# ================================================================ 1. what, how, weights
s = new_slide('What the DII is, how it is computed, and its weights', notes=(
    'Intuitive = the car does what an occupant expects a competent human driver to do, and never feels unsafe. '
    'Every controller drives the same scenarios; six raw metrics are mapped to 0-100 and weighted. '
    'All numbers were recomputed by replaying each controller in the repo simulator; each TD3 checkpoint runs in its own commit’s code. '
    'Weights are expert judgement, set before seeing results: safety and human-likeness lead at 0.30 each. '
    'The comfort group keeps 4:3:3:2 proportions; jerk leads it as the best single predictor of rated smoothness; oscillation is smallest because it overlaps with jerk and acceleration. '
    'Energy, speed and throughput are left out on purpose: they measure efficiency, not how intuitive the ride feels.'))
tb(s, L, Inches(1.1), W, Inches(0.45), [('Intuitive = the car does what an occupant expects a competent human driver to do, and never feels unsafe. The DII turns that into one 0–100 score from six measurable parts of the ride.', {'size': 11})])
steps = [('1  Simulate', '15-car ring road, dt 0.1 s. 7 benchmark scenarios + 20 unseen lead profiles.'),
         ('2  Measure', 'Per follower: TTC, gap, jerk, accel, headway, reversals, error vs a human model.'),
         ('3  Normalise', 'Each metric → 0–100 on a line between a “good” and a “bad” anchor.'),
         ('4  Weight', 'Weighted sum of six sub-scores. A collision zeroes S and halves the DII.'),
         ('5  Average', 'Mean over the 5 dynamic benchmark scenarios (or 20 unseen profiles).')]
cw, gap = Inches(1.66), Inches(0.175)
for i, (h, b) in enumerate(steps):
    x = L + (cw + gap) * i; dark = i == 3
    card(s, x, Inches(1.62), cw, Inches(1.18), NAVY if dark else LIGHT, f'Step {i+1}')
    tb(s, x + Inches(0.1), Inches(1.67), cw - Inches(0.2), Inches(0.28), [(h, {'size': 11, 'bold': True, 'color': MINT if dark else TEAL})])
    tb(s, x + Inches(0.1), Inches(1.95), cw - Inches(0.2), Inches(0.82), [(b, {'size': 9.5, 'color': WHITE if dark else NAVY})])
card(s, L, Inches(2.95), Inches(5.55), Inches(2.0), LIGHT, 'Formula card')
tb(s, L + Inches(0.18), Inches(3.0), Inches(5.2), Inches(1.9), [
    ('DII = γ × (0.30·S + 0.30·Hm + 0.133·J', {'size': 12.5, 'bold': True, 'font': 'Cambria'}),
    ('            + 0.10·A + 0.10·H + 0.067·O),   γ = 0.5 if any collision', {'size': 12.5, 'bold': True, 'font': 'Cambria', 'after': 4}),
    ('lin(x; good, bad) = 100 × clip((x − bad) / (good − bad), 0, 1)', {'size': 10, 'font': 'Cambria', 'color': GREY, 'after': 5}),
    ([('Why these weights: ', {'bold': True, 'color': TEAL}), ('safety and human-likeness lead, because intuitive means acting as an occupant expects a human to, without feeling unsafe. Jerk leads the comfort group; oscillation is smallest because it overlaps jerk and acceleration.', {})], {'size': 10})], MSO_ANCHOR.MIDDLE)
bar_chart(s, Inches(6.0), Inches(2.9), Inches(3.5), Inches(2.1), ['Safety (S)', 'Human-like (Hm)', 'Jerk (J)', 'Accel. (A)', 'Time-gap (H)', 'Oscillation (O)'],
          (0.30, 0.30, 0.133, 0.10, 0.10, 0.067), 'Weights (sum = 1)', [NAVY, NAVY, TEAL, TEAL, TEAL, TEAL], '0.0##', 0.38, 'Weights chart')
footnote(s, 'Energy, speed and throughput are left out on purpose: they measure efficiency, not how intuitive the ride feels.')

# ================================================================ 2. six formulas
s = new_slide('The six component formulas', notes=(
    'Each card: the raw metric, the 0-100 score, and where the anchors come from. Followers i = 1..14, samples every 0.1 s, pooled over one scenario.'))
FM = {'size': 9.5, 'font': 'Cambria'}
forms = [('S  Safety margin', '0.30', ['TTC = gap / closing speed;  TET = s/car/min with TTC < 3 s', 'S = ½ lin(TET; 0, 6) + ½ lin(g_min; 6 m, 1 m)'], 'TTC < 3 s = conflict; 6 m = project min_dis. Collision → S = 0.'),
         ('Hm  Human-likeness', '0.30', ['aʰ = Improved-IDM accel in the same v, gap, Δv', 'Hm = lin(√mean(a − aʰ)²; 0.2, 1.5)  [m/s²]'], '0.2 m/s² ≈ human noise; 1.5 = clearly different.'),
         ('J  Jerk', '0.133', ['j = Δa / Δt;  J_rms = √mean(j²);  J₉₅ = p95 |j|', 'J = ½ lin(J_rms; 0.3, 2.0) + ½ lin(J₉₅; 0.9, 4.0)'], '0.3 barely felt, 0.9 comfortable, ≥2 m/s³ uncomfortable.'),
         ('A  Acceleration comfort', '0.10', ['A_rms = √mean(a²);  f_out = share outside [−3, +2] m/s²', 'A = ½ lin(A_rms; 0.3, 1.5) + ½ lin(f_out; 0, 0.05)'], 'Band from ACC practice (ISO 15622) and hard-braking studies.'),
         ('H  Time-gap consistency', '0.10', ['THW = gap / v (v > 3 m/s);  CV = std / mean per car', 'H = ½ lin(CV; 0.1, 0.6) + ½ lin(share in 0.8–3 s; 1, 0.5)'], 'Humans hold a steady 1–2 s gap; < 0.8 s is tailgating.'),
         ('O  Oscillation and effort', '0.067', ['R = reversals/min;  SW = max decel / leader decel;  E = |a| / |a_lead|', 'O = ⅓[lin(R; 2, 20) + lin(SW; 1, 2) + lin(E; 1, 2.5)]'], 'SW = 1 means string stable; R catches pedal hunting.')]
cw, ch_ = Inches(4.425), Inches(1.17)
for i, (h, w, lines, why) in enumerate(forms):
    x = L + (cw + Inches(0.15)) * (i % 2); y = Inches(1.17) + (ch_ + Inches(0.1)) * (i // 2)
    top = i < 2
    card(s, x, y, cw, ch_, NAVY if top else LIGHT, f'Formula {i}')
    tb(s, x + Inches(0.13), y + Inches(0.07), cw - Inches(0.26), ch_ - Inches(0.1), [
        ([(h, {'bold': True, 'color': MINT if top else TEAL}), (f'   weight {w}', {'bold': True, 'color': WHITE if top else NAVY})], {'size': 11, 'after': 3}),
        (lines[0], dict(FM, color=WHITE if top else NAVY, after=1)),
        (lines[1], dict(FM, color=WHITE if top else NAVY, after=3)),
        (why, {'size': 9.5, 'italic': True, 'color': BORDER if top else GREY})])
footnote(s, 'Special cases: if the leader never brakes, the SW term is 100; if it barely moves, E scores the followers’ absolute mean |a| (0.05 → 0.5 m/s²).', Inches(5.1))

# ================================================================ 4. benchmark
s = new_slide('Benchmark: TD3_56 is level with ACC+BCC', notes=(
    'Mean DII over the 5 dynamic benchmark scenarios (steady cruise is ~100 for all). 0 collisions for every controller. '
    'Per scenario: TD3_56 best in stop-hold (70.7) and sine wave (97.9); ACC+BCC best in stop-and-go (83.5); ACC best in emergency braking (66.0); BCC only worst in emergency braking (38.2, 0.9 m gap). '
    'TD3_56: most human-like (Hm 88), joint-best on safety, loses on jerk spikes in stop-and-go (p95 2.6 vs 1.0 m/s3). BCC only: smoothest but least safe and least human-like. ACC: harshest ride (A 23), amplifies braking waves 1.85x. '
    'IDM: s* = s0 + max(0, vT + v dv / 2 sqrt(ab)); a_h = a(1 - (s*/gap)^2).'))
bar_chart(s, L, Inches(1.15), Inches(3.6), Inches(3.75), ['Human model (IDM)', 'ACC+BCC', 'TD3_56 (RL)', 'ACC', 'BCC only'],
          (90.0, 79.4, 79.1, 73.6, 72.4), 'Benchmark DII (0–100)', [GREY, NAVY, TEAL, NAVY, NAVY], '0.0', 110, 'DII chart')
rows = [['Controller', 'S', 'Hm', 'J', 'A', 'H', 'O'],
        ['Human model', 87, '100*', 94, 65, 85, 96],
        ['ACC+BCC', 74, 84, 90, 42, 94, 93],
        ['TD3_56', 76, 88, 74, 46, 93, 91],
        ['ACC', 76, 77, 76, 23, 93, 88],
        ['BCC only', 66, 58, 97, 62, 96, 97]]
table(s, Inches(4.3), Inches(1.2), Inches(5.2), rows, [Inches(1.3)] + [Inches(0.65)] * 6, size=10.5,
      bold_cells={(3, 1), (3, 2), (5, 3), (5, 4), (5, 5), (5, 6), (4, 1)}, row_h=Inches(0.31))
card(s, Inches(4.3), Inches(3.2), Inches(5.2), Inches(1.75), NAVY, 'IDM card')
tb(s, Inches(4.45), Inches(3.25), Inches(4.9), Inches(1.65), [
    ('Why compare with the Intelligent Driver Model (IDM)?', {'size': 11, 'bold': True, 'color': MINT, 'after': 3}),
    ('IDM is the standard model of a human following the car ahead (Treiber 2000; improved form 2013). The repo has no recorded human driving, so IDM stands in: Hm measures how close each controller’s acceleration is to what a human would do in the same state.', {'size': 9.5, 'color': WHITE, 'after': 3}),
    ([('Settings: ', {'bold': True, 'color': MINT}), ('a = 1.4, b = 2.0 m/s² (human comfort); s0 = 6 m, T = 0.8 s (project spacing). ', {}), ('It scores 90, not 100: ', {'bold': True, 'color': MINT}), ('a harsh leader forces even a human into hard braking, so 90 is the realistic ceiling.', {})], {'size': 9.5, 'color': WHITE})], MSO_ANCHOR.MIDDLE)
footnote(s, '*Hm is 100 for the human model by construction; read that row as a ceiling. 0 collisions for every controller on the benchmark.')

# ================================================================ 5. inference
s = new_slide('Which controller is better today, and why', notes=(
    'Unseen profiles: 20 seeded random 60 s leader schedules (holds, brakes, full stops, accelerations up to 25 m/s). '
    'TD3_56 minus ACC+BCC there: -4.3 points, 95% bootstrap CI [-8.6, -0.6]. TD3_56 collided once (25 to 0 m/s stop in 3.9 s, ~1 m/s creep). '
    'Note: the classical min gaps on the earlier TD3_49 benchmark slide come from the training log and do not reproduce when re-run (e.g. ACC+BCC emergency braking 2.3 m, not 9.60 m).'))
blocks = [('ACC  ·  73.6', 'Looks only at the car ahead, with fixed gains.', 'Corrects hard and passes braking waves down the platoon: safe and fairly human-like, but uncomfortable.'),
          ('BCC only  ·  72.4', 'Balances the gap ahead against the gap behind.', 'The averaging damps waves (smoothest), but it brakes late for the car ahead, and no human drives for the car behind.'),
          ('ACC+BCC  ·  79.4', 'Integration factor blends ACC and BCC by situation.', 'Keeps most of BCC’s smoothness and ACC’s forward safety: the most balanced and the most robust.'),
          ('TD3_56 (RL)  ·  79.1', 'Learned from comfort, safety and string-stability rewards.', 'Most human-like and best at absorbing waves. Jerk spikes and tight emergency gaps remain from the TD3_49/50 blend.')]
for i, (h, what, why) in enumerate(blocks):
    x = L + Inches(4.575) * (i % 2); y = Inches(1.17) + Inches(1.27) * (i // 2)
    dark = i == 3
    card(s, x, y, Inches(4.425), Inches(1.17), NAVY if dark else LIGHT, f'Inference {i}')
    tb(s, x + Inches(0.15), y + Inches(0.07), Inches(4.13), Inches(1.06), [
        (h, {'size': 12, 'bold': True, 'color': MINT if dark else TEAL, 'after': 2}),
        (what, {'size': 10, 'bold': True, 'color': WHITE if dark else NAVY, 'after': 1}),
        (why, {'size': 10, 'color': WHITE if dark else NAVY})])
stats = [('85.1 vs 80.7', 'ACC+BCC vs TD3_56 on 20 unseen profiles (gap holds statistically)'),
         ('1 collision', 'TD3_56 on a 25 → 0 m/s stop; ACC and ACC+BCC had none')]
for i, (v, lab) in enumerate(stats):
    x = L + Inches(2.3) * i
    tb(s, x, Inches(3.78), Inches(2.2), Inches(0.42), [(v, {'size': 18, 'bold': True, 'font': 'Georgia', 'color': TEAL})], MSO_ANCHOR.MIDDLE)
    tb(s, x, Inches(4.2), Inches(2.1), Inches(0.6), [(lab, {'size': 9.5, 'color': GREY})])
card(s, Inches(5.1), Inches(3.75), Inches(4.4), Inches(1.15), NAVY, 'Verdict')
tb(s, Inches(5.27), Inches(3.82), Inches(4.06), Inches(1.0), [
    ('Verdict: ACC+BCC, narrowly', {'size': 12, 'bold': True, 'color': MINT, 'after': 3}),
    ('TD3_56 ties it on the benchmark and is the most human-like, but generalises less well. Restoring TD3_49’s safety margin while keeping TD3_56’s smoothness should put RL ahead.', {'size': 10, 'color': WHITE})], MSO_ANCHOR.MIDDLE)
footnote(s, 'Benchmark DII shown after each controller name. Unseen profiles: seeded random leader schedules with holds, brakes, full stops and accelerations.', Inches(5.1))

prs.save(DST)
print('saved', len(prs.slides))
