"""Build the illustrated teaching report without loading any game models."""
from pathlib import Path
import math

from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/splendor_players_intro_20261023.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
for name, file in [("Body", "calibri.ttf"), ("Bold", "calibrib.ttf"),
                   ("Light", "calibril.ttf"), ("Italic", "calibrii.ttf")]:
    pdfmetrics.registerFont(TTFont(name, "C:/Windows/Fonts/" + file))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Italic", boldItalic="Bold")
W, H = 595.276, 841.89
M, CW = 43, W - 86
NAVY = HexColor("#18344A")
INK = HexColor("#263D4D")
MUTED = HexColor("#526976")
TEAL = HexColor("#087F87")
GOLD = HexColor("#B76D22")
CORAL = HexColor("#B94D47")
PALE = HexColor("#F0F5F6")
LINE = HexColor("#CFDBDF")
BG = HexColor("#FCFCF9")
c = canvas.Canvas(str(OUT), pagesize=(W, H), pageCompression=1)
c.setTitle("Two ways to think ahead in Splendor")
c.setAuthor("Splendor research programme")
c.setSubject("A simple illustrated comparison of our verified planner and adapted SplendoRust")

def rect(x, y, w, h, fill=PALE, stroke=None, radius=9):
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.roundRect(x, H-y-h, w, h, radius, fill=1, stroke=int(stroke is not None))

def text(x, y, s, size=11.5, font="Body", color=INK):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, H-y-size*0.82, s)

def para(s, x, y, w=CW, size=11.5, leading=15.5, color=INK,
         font="Body", max_h=None):
    style = ParagraphStyle("p", fontName=font, fontSize=size, leading=leading,
                           textColor=color, spaceAfter=0, allowWidows=0, allowOrphans=0)
    p = Paragraph(s, style)
    _, h = p.wrap(w, 900)
    if max_h is not None and h > max_h + .1:
        raise ValueError(f"Text exceeds box: {h} > {max_h}: {s[:65]}")
    if y+h > 782:
        raise ValueError(f"Text exceeds page: {y+h}: {s[:65]}")
    p.drawOn(c, x, H-y-h)
    return y+h

def arrow(x1, y1, x2, y2, color=MUTED, width=1.4):
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(width)
    c.line(x1, H-y1, x2, H-y2)
    a = math.atan2(y2-y1, x2-x1)
    p = c.beginPath(); p.moveTo(x2, H-y2)
    for z in [a+2.65, a-2.65]:
        p.lineTo(x2+6*math.cos(z), H-(y2+6*math.sin(z)))
    p.close(); c.drawPath(p, fill=1, stroke=0)

def circle(x, y, r=9, color=TEAL):
    c.setFillColor(color); c.circle(x, H-y, r, fill=1, stroke=0)

def card(x, y, label, caption, width=116, height=86, color=TEAL):
    rect(x,y,width,height,white,LINE)
    circle(x+16,y+18,5,color)
    text(x+29,y+12,label,11,"Bold",color)
    para(caption,x+12,y+37,width-24,10,13,max_h=height-42)

def note(y, title, body, color=TEAL, h=83):
    rect(M,y,CW,h,PALE)
    c.setFillColor(color); c.rect(M,H-y-h,4,h,fill=1,stroke=0)
    text(M+15,y+12,title,12,"Bold",color)
    para(body,M+15,y+34,CW-30,11,14,max_h=h-40)

def page(n, tag, title, subtitle):
    c.setFillColor(BG); c.rect(0,0,W,H,fill=1,stroke=0)
    text(M,27,"SPLENDOR  /  AN INTRODUCTION",9,"Bold",TEAL)
    text(W-177,27,"23 OCTOBER 2026",9,"Body",MUTED)
    text(M,65,tag.upper(),10,"Bold",GOLD)
    para(title,M,88,CW,28,31,NAVY,"Bold",max_h=66)
    para(subtitle,M,161,CW,12,16,MUTED,max_h=48)
    c.setStrokeColor(LINE); c.setLineWidth(.6); c.line(M,47,W-M,47)
    text(M,803,"Verified local versions  |  Introductory guide",9,"Body",MUTED)
    text(W-M-26,803,f"{n} / 6",9,"Bold",TEAL)

def end(): c.showPage()

# PAGE 1: give the reader a concrete vocabulary before comparing architectures.
page(1,"Start here","Two ways to think ahead\nin Splendor".replace("\n","<br/>"),
     "How our strongest verified local planner and adapted SplendoRust decide what to play.")
para("In Splendor, you collect gems, buy cards and race for prestige points. Cards can also make later purchases cheaper. The hard question is whether a move that helps later will pay off before the opponent finishes the game.",M,222,size=12,leading=17)
text(M,302,"A complete player has three jobs",17,"Bold",NAVY)
card(M,337,"1  Observe","Read the board and the information this player is allowed to know.",151,116)
card(M+179,337,"2  Think","Use a learned model and a planning procedure to compare moves.",151,116)
card(M+358,337,"3  Act","Return one legal, complete move before the clock expires.",151,116)
arrow(M+154,394,M+173,394); arrow(M+333,394,M+352,394)
rect(M,480,245,150,white,LINE); rect(M+264,480,245,150,white,LINE)
text(M+16,496,"MODEL",12,"Bold",TEAL)
para("Patterns learned earlier",M+16,521,213,16,20,NAVY,"Bold")
para("Like a player's experience: which moves look promising, and sometimes how good a position looks.",M+16,572,213,11.5,15.5,max_h=50)
text(M+280,496,"SEARCH",12,"Bold",GOLD)
para("Thinking before this move",M+280,521,213,16,20,NAVY,"Bold")
para("Like looking ahead: try possible moves and replies, then decide where to spend more thought.",M+280,572,213,11.5,15.5,max_h=50)
note(657,"The comparison is between whole players",
     "A model is one component. The 60-game match tested each complete decision system: its model, search, rule adapter and time handling together.",h=87)
end()

# PAGE 2: the root shortlist and full-game continuation policy.
page(2,"Our verified planner","Choose a few moves,<br/>then play out their futures",
     "Its central question: which shortlisted move wins most often in the imagined games? [1, 2]")
text(M,221,"1  Build a shortlist of up to four distinct moves",15,"Bold",NAVY)
labels=[("A","Raw035's first choice"),("B","Raw035's second choice"),
        ("C","Seeded exploration"),("D","Learned proposal")]
for i,(label,desc) in enumerate(labels):
    card(M+i*130,254,label,desc,119,82,TEAL if i<2 else GOLD)
para("Raw035 is a neural action scorer. A second learned scorer proposes an extra move. The exploration slot follows a fixed sampling rule. The shortlist can be smaller if fewer distinct moves exist.",M,355,size=11.5,leading=15.5)
text(M,422,"2  Test each candidate in the same 128 possible worlds",15,"Bold",NAVY)
rect(M,455,CW,137,white,LINE)
for row,label in enumerate("ABCD"):
    y=476+row*26
    text(M+14,y-4,label,12,"Bold",TEAL)
    for col in range(6):
        x=M+46+col*43
        rect(x,y-8,28,16,HexColor("#D4E9E7"),radius=3)
    text(M+313,y-5,"...  128 worlds",10.5,"Body",MUTED)
    arrow(M+397,y,M+423,y)
    text(M+432,y-5,"score",10.5,"Bold",TEAL)
para("A world is one plausible arrangement of hidden cards. Raw035 plays both sides greedily, taking its highest-scored legal move each turn. Play continues to the end; rare forced-pass deadlocks use a defined score-based convention.",M,609,size=11.5,leading=15.5)
note(675,"3  Pick the highest average outcome",
     "A win counts +1, a draw 0 and a loss -1. Up to 4 x 128 = 512 continuations inform one real move. These forecasts depend on how realistically Raw035 plays both sides.",h=89)
end()

# PAGE 3: adaptive tree search is different from independent full rollouts.
page(3,"Verified SplendoRust","Explore a tree,<br/>guided by a neural network",
     "Its central question: which move deserves more investigation, and what do the searches say? [3, 4]")
para("SplendoRust considers the eligible legal main moves. Its network suggests promising moves (policy) and estimates how favorable a position looks (value). The search, written in Rust, explores moves and replies and asks the network for guidance.",M,222,size=11.5,leading=16)
rect(M,293,CW,241,white,LINE)
circle(210,321,12,NAVY); text(232,315,"Current position",11,"Bold",NAVY)
for x in [112,210,308]: arrow(210,335,x,373,TEAL,1.4 if x!=210 else 3.0)
for x,label in [(112,"Buy"),(210,"Take gems"),(308,"Reserve")]:
    circle(x,386,10,TEAL); text(x-22,403,label,10,"Bold",TEAL)
for x,xx in [(112,84),(112,139),(210,183),(210,237),(308,281),(308,335)]:
    arrow(x,426,xx,450,LINE if x!=210 else TEAL,1 if x!=210 else 2.8)
    circle(xx,460,7,TEAL if x==210 else HexColor("#BCD4D5"))
text(M+15,498,"Illustration only: the highlighted branch receives more searches.",9.5,"Italic",MUTED)
rect(378,334,157,149,PALE)
text(391,350,"NETWORK GUIDANCE",9,"Bold",TEAL)
para("<b>Policy</b><br/>Which moves look promising?<br/><br/><b>Value</b><br/>How good does this position look?",391,375,130,10.5,14,max_h=104)
text(M,554,"Repeat 6,400 search simulations",16,"Bold",NAVY)
para("Each simulation follows part of the tree, judges a position, and updates the visited branches. PUCT balances promising moves with moves that need more checking. For an ordinary main move, the most visited branch wins; the network's initial preference breaks ties.",M,585,size=11.5,leading=16)
note(667,"A simulation is not a complete game",
     "6,400 tree simulations and our 512 full continuations are different units of work. SplendoRust can judge a position before the imagined game has ended, using its learned value estimate.",h=91)
end()

# PAGE 4: explain the common neural idea and concrete differences.
page(4,"Inside the models","Both learn patterns.<br/>They use them differently.",
     "Training changes a network's numerical weights before play. In these benchmarks, the weights stay fixed throughout every game.")
text(M,222,"What does an attention network do?",16,"Bold",NAVY)
para("It combines relevant relationships: for example, a card's cost with a player's gems and discounts. Our models compare legal moves in the context of the board. SplendoRust compares board entities, such as cards and players. Below is a simplified picture of that second approach.",M,252,size=11.5,leading=16)
for i,(t,d) in enumerate([("Cards","Costs and points"),("Players","Gems and discounts"),("Market","Available options")]):
    rect(M+i*173,329,163,57,white,LINE)
    text(M+12+i*173,340,t,12,"Bold",TEAL)
    text(M+12+i*173,362,d,9.8,"Body",MUTED)
    arrow(M+81+i*173,390,297,415,LINE)
rect(160,418,275,44,TEAL)
text(188,431,"Attention combines relationships",12,"Bold",white)
arrow(297,465,169,490); arrow(297,465,423,490)
text(97,497,"Move preferences",12,"Bold",TEAL)
text(350,497,"Position estimates",12,"Bold",GOLD)
rows=[
    ("Role in the player","Our verified planner","Verified SplendoRust"),
    ("Model structure","Raw035 and a proposal model. Each has 2 attention layers: stages of processing.","One Entity Transformer with 6 attention layers and about 4.78 million learned numbers."),
    ("How it reads the game","295 numbers describe what the acting player knows; 79 numbers describe each legal move.","31 pieces of information, such as cards and players; 48 numbers describe each piece."),
    ("What the planner uses","Move scores for proposals and simulated play. Its position-value outputs are unused.","Move preferences (policy) and position estimates (value) guide the tree search."),
]
widths=[98,202,209]; heights=[30,67,59,66]
y=531
for j,row in enumerate(rows):
    x=M
    for i,cell in enumerate(row):
        rect(x,y,widths[i],heights[j],NAVY if j==0 else (PALE if j%2 else white),radius=0)
        para(cell,x+9,y+8,widths[i]-18,9.5,12,white if j==0 else INK,"Bold" if j==0 or i==0 else "Body",max_h=heights[j]-12)
        x+=widths[i]
    y+=heights[j]
para("Our proposal models imitate historical planner choices. We reused SplendoRust's published weights. The three local fits play separately; they do not vote together. [1, 3, 4]",M,758,CW,8.5,10.5,MUTED,max_h=22)
end()

# PAGE 5: distinguish observed strength from causal interpretation.
page(5,"What the games taught us","A large strength gap.<br/>A narrower explanation.",
     "60 games, in 30 swapped-seat pairs, with 120 seconds per move for both players. Accepted recovered benchmark. [5]")
rect(M,224,245,102,TEAL); rect(M+264,224,245,102,NAVY)
text(M+18,240,"60 WINS",30,"Bold",white)
text(M+18,288,"Verified SplendoRust",12,"Body",white)
text(M+282,240,"0 WINS",30,"Bold",white)
text(M+282,288,"Our verified planner",12,"Body",white)
para("Each local proposal fit lost all 20 games: 59 ordinary losses and one timeout overall. One interrupted move was retried under a disclosed recovery amendment; the genuine timeout loss remained. The result compares complete players, not the network and search separately.",M,344,size=11.5,leading=16)
text(M,423,"What the postmortem observed",16,"Bold",NAVY)
rect(M,456,CW,108,white,LINE)
text(M+14,470,"Average final prestige in the 59 natural games",10.5,"Bold",MUTED)
for label,value,yy,color in [("Our player",7.03,502,GOLD),("SplendoRust",15.63,536,TEAL)]:
    text(M+14,yy-3,label,11,"Body",INK)
    rect(M+112,yy-5,value*18,15,color,radius=3)
    text(M+122+value*18,yy-5,f"{value:.2f}",11,"Bold",color)
para("Our player bought more cards but scored more slowly. Its imagined games also tended to last longer than the real games. That supports a hypothesis: it may invest in plans that look good against a slow simulated opponent but mature too late against SplendoRust. [2]",M,583,size=11.5,leading=16)
note(665,"Keep observation and explanation separate",
     "We have not proved that stronger simulated opponents fix the problem. The recent 64-world study passed timing and replay checks, then stopped because independent assessment was projected to exceed its time cap. [6]",h=95)
end()

# PAGE 6: recommended next research step and human-readable source key.
page(6,"Where we go next","Build from the stronger<br/>verified player",
     "Keep the existing SplendoRust version as the reference. Improve one component, then test the complete player again.")
text(M,222,"First proposed study: does more search help?",16,"Bold",NAVY)
rect(M,254,210,73,white,LINE); rect(M+299,254,210,73,white,LINE)
text(M+16,267,"REFERENCE",10,"Bold",TEAL)
text(M+16,288,"6,400 simulations",19,"Bold",NAVY)
text(M+315,267,"ONE CANDIDATE",10,"Bold",GOLD)
text(M+315,288,"10,240 simulations",19,"Bold",NAVY)
arrow(M+222,290,M+286,290)
para("Keep the network, rules, hidden-world sampling and all other search settings fixed. Give both players 120 seconds per complete move. This asks whether more thinking helps before paying for new training or a redesigned planner.",M,347,size=11.5,leading=16)
text(M,414,"The proposed sequence",14,"Bold",NAVY)
steps=[("1","Qualify","Correct moves and privacy; every test decision must finish within 96 seconds."),
       ("2","Check operation","Six practice games, excluded from the strength result."),
       ("3","Measure strength","60 fresh paired games against the unchanged reference."),
       ("4","Confirm","A promising screen needs independent confirmation and other opponents.")]
for i,(n,title,body) in enumerate(steps):
    y=447+i*49; circle(M+11,y+11,11,TEAL)
    text(M+8,y+5,n,11,"Bold",white)
    text(M+32,y-1,title,11.5,"Bold",NAVY)
    para(body,M+148,y-1,CW-148,10.5,13,max_h=30)
para("The study is proposed, not launched. More search may bring little benefit, and it still reuses the same three plausible hidden-card arrangements sampled for each move. Larger-data training stays deferred.",M,650,size=10.5,leading=14)
text(M,707,"Sources and scope",10,"Bold",TEAL)
para("[1] Corrected planner qualification and frozen model registry, 18 Oct.  [2] Saved-game postmortem, 22 Oct.  [3] SplendoRust 120-second qualification, 18 Oct.  [4] Pinned SplendoRust source and public champion model, commit 1828507.  [5] Accepted amended benchmark finalization, 22 Oct.  [6] A64/B64 qualification and throughput hold, 23 Oct. Full paths, identities and study gates are in the companion roadmap.",M,725,CW,8.8,11.5,MUTED,max_h=50)
c.linkURL("https://github.com/paytonjjones/splendorust/blob/1828507b8d2f0e02bc03d102b9c6d29a54fa8d57/docs/CHAMPION.md",(M,H-777,W-M,H-725),relative=0,thickness=0)
end()
c.save()
print(OUT)
