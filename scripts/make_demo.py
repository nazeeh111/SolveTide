"""Caption actual frozen-source UI captures, with edited pauses explicitly labeled.

No field interpolation, invented runtime, synthetic cursor or simulated browser.
The five raw screenshots and AX observations remain editable source assets.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import os
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"solvetide-mpl"))
import matplotlib
ROOT=Path(__file__).resolve().parents[1]
FONT=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
SERIF=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSerif.ttf'
SCENES=[
 ('final-01-initial.jpg','1  Start with heat','Actual assembled field.','Forward and adjoint checked.'),
 ('final-02-local-edit.jpg','2  Edit local material','Contrast increases by 5%.','New operator. New solution.'),
 ('final-03-remesh.jpg','3  Change the mesh','Old mappings are invalidated.','Recompute on the new grid.'),
 ('final-04-failure.jpg','4  Exhaust the budget','One iteration is insufficient.','No accepted field or gradient.'),
 ('final-05-direct-fallback.jpg','5  Recompute directly','Fresh sparse-direct fallback.','Original equations checked.')]
OUT=ROOT/'docs/demo.next.mp4'
with tempfile.TemporaryDirectory(prefix='solvetide-demo-') as tmp:
    tmp=Path(tmp);clips=[]
    for i,(name,title,line1,line2) in enumerate(SCENES):
        clip=tmp/f'{i}.mp4';clips.append(clip)
        # drawtext arguments never contain user input. Quotes/colons are absent.
        vf=(f"scale=-2:828,pad=1440:960:40:92:color=0xf5f4ed,"
            f"drawtext=expansion=none:fontfile='{FONT}':text='SOLVETIDE':x=820:y=100:fontsize=25:fontcolor=0x183b3b,"
            f"drawtext=expansion=none:fontfile='{SERIF}':text='{title}':x=820:y=230:fontsize=31:fontcolor=0x183b3b,"
            f"drawtext=expansion=none:fontfile='{FONT}':text='{line1}':x=820:y=315:fontsize=20:fontcolor=0x32665e,"
            f"drawtext=expansion=none:fontfile='{FONT}':text='{line2}':x=820:y=355:fontsize=20:fontcolor=0x32665e,"
            f"drawtext=expansion=none:fontfile='{FONT}':text='Actual browser captures':x=820:y=600:fontsize=18:fontcolor=0x617372,"
            f"drawtext=expansion=none:fontfile='{FONT}':text='Edited pauses. No real-time claim.':x=820:y=635:fontsize=16:fontcolor=0x617372,"
            f"drawtext=expansion=none:fontfile='{FONT}':text='Discrete checks do not certify physical truth.':x=820:y=700:fontsize=15:fontcolor=0x617372")
        subprocess.run(['ffmpeg','-y','-loglevel','error','-loop','1','-i',str(ROOT/'docs/demo'/name),'-vf',vf,
                        '-t','3','-r','24','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-an',str(clip)],check=True)
    concat=tmp/'sequence.txt';concat.write_text(''.join(f"file '{p}'\n" for p in clips))
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(OUT)],check=True)
OUT.replace(ROOT/'docs/demo.mp4')
print(ROOT/'docs/demo.mp4')
