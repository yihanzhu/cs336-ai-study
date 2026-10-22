import hashlib,json,pathlib
root=pathlib.Path(__file__).parent.parent;records=[]
for p in sorted((root/'work/alignment').glob('20*.json')):
 d=json.loads(p.read_text())
 rows=[]
 for w in d['windows']:
  rows.append({'start':w['start'],'end':w['end'],'segments_accounted':w['segments'],'topic_terms':w['topics'][:7],
   'material_navigation_candidates':[{'page':x['page'],'lexical_score':x['lexical_score']} for x in w['material_candidates']],
   'classroom_navigation_candidates':[{'type':x['type'],'start':x['start'],'end':x['end']} for x in w['review_candidates']],
   'status':'automated_navigation_not_semantic_acceptance'})
 records.append({'year':d['year'],'playlist_order':d['lecture'],'video_id':d['video_id'],'video_url':d['video_url'],'source_type':d.get('source_type','uploaded_caption'),'windows':rows})
out={'purpose':'Navigation for every retrieved caption segment; keyword and marker matches are candidates only, not confirmed additional content or student questions. No raw transcript text is distributed.','method':'Fixed 150-second bins; lexical terms and slide/page candidates; sentence markers do not prove novelty or correctness.','transcripts':records,'window_count':sum(len(x['windows']) for x in records),'segments_accounted':sum(w['segments_accounted'] for x in records for w in x['windows'])}
assert out['window_count']==1131 and out['segments_accounted']==56527
(root/'outputs/CS336-课堂片段索引.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print('Navigation index:',out['window_count'],'windows;',out['segments_accounted'],'caption segments; no transcript text.')
