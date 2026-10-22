"""Create ignored local progress from a blank template without overwriting learner data."""
import argparse,pathlib

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--language',choices=['en','zh'],default='en');a=p.parse_args()
 root=pathlib.Path(__file__).resolve().parent.parent;dest=root/'local/learning-progress.md'
 if dest.exists():
  print('Existing local/learning-progress.md preserved; nothing overwritten.');return
 source=root/'outputs'/('learning-progress-template.en.md' if a.language=='en' else '学习进度模板.md')
 content=source.read_text();dest.parent.mkdir(parents=True,exist_ok=True)
 try:
  with dest.open('x') as f:f.write(content)
 except FileExistsError:
  print('Existing local/learning-progress.md preserved; nothing overwritten.');return
 dest.chmod(0o600);print('Created local/learning-progress.md; local only, not for Git or release archives.')

if __name__=='__main__':main()
