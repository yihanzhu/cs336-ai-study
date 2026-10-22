"""Append authored audit additions with precise provenance. Called during unified build."""
import json
from batch_audio_notes import BATCH_AUDIO_NOTES
from demo_visual_notes import DEMO_VISUAL_NOTES
from trace_figure_notes import TRACE_FIGURE_NOTES
from trace_value_notes import TRACE_VALUE_NOTES
from full_pdf_notes import PDF_FULL_NOTES
from assignment_figure_notes import ASSIGNMENT_FIGURE_NOTES

def append_audit(lines,root,video_link,by_video):
 lines += ['## 十、原始音频比对与问答补核','', '35场可取得录像均有原始英文音频证据。34场用完整base.en转写比对上传字幕，再对所有字幕INAUDIBLE及低相似度/低置信度窗口用small.en转写；L08已完整双模型处理。下列为对疑难文本的语义检查，不是人工逐句听校。两模型同属Whisper家族，可能共同错听；无法恢复的原问保留缺口。完整音频与转写留在本地，分享包仅含原创说明与源哈希。','']
 for k,n in sorted(BATCH_AUDIO_NOTES.items()):
  y,num=map(int,k.split('-'));v=by_video[(y,num)]
  lines += [f'### {k} · 音频补核','']
  for a,b,t in n['findings']:lines += [f'- {video_link(v,a,b)} · {t}']
 lines += ['','## 十一、补取的演示画面','', '追加136帧均已查看。取点来自互动/演示候选和全文校读笔记；覆盖重要变化，不声称遍历每个视频帧。有些帧只拍到讲者，不能互证其描述的屏幕。完整帧哈希及来源见逐讲对照记录。','']
 for k,n in sorted(DEMO_VISUAL_NOTES.items()):
  y,num=map(int,k.split('-'));v=by_video[(y,num)]
  lines += [f'- {k} · '+ '、'.join(video_link(v,t) for t in n['frames'])+'：'+n['finding']]
 lines += ['','## 十二、可执行讲义的图与数组','', '312个固定仓库版本的图像来源已取得，按内容哈希去重为231张，全部查看版面和主图；两张GIF另展开全部266帧核对动态步骤。序号以《CS336-讲义图像核验.json》为准。图中小字、所有原始曲线点及所引实验不逐项独立复现。405个复合输出完成有限值/已记录shape检查，另重算BPE、softmax、逐行和、LSH及推理统计等关系；随机权重输出和GPU计时仍是作者缓存。','']
 for a,b,t in TRACE_FIGURE_NOTES:lines += [f'- 图像 {a}–{b}：{t}']
 lines += ['','数组和记录的教学处理：','']
 for f,t,s in TRACE_VALUE_NOTES:lines += [f'- {f} · {t}：{s}']
 lines += ['','## 十三、其余讲义与作业逐页图表核对','', '以下范围覆盖其余24份核心PDF的1366页与17份作业的450页；连同第九节L08的73页，共42份1889页。每页版面/主图均已看过。作业的密集文本按所列公式重点交叉读，未声称逐字解决所有题目。若本节纠错与前文初轮课堂摘点不同，以本节明确列出的原始依据和限定为准。','']
 for file,n in {**PDF_FULL_NOTES,**ASSIGNMENT_FIGURE_NOTES}.items():
  lines += [f'### {file}','']
  for a,b,t in n['findings']:lines += [f'- 第{a}–{b}页：{t}']
  if n.get('primary_sources'):lines += ['','原始依据：'+' · '.join(f'[来源{i+1}]({u})' for i,u in enumerate(n['primary_sources']))]
  lines += ['']
