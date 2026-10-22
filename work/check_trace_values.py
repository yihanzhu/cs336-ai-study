import ast,collections,hashlib,json,math,pathlib
W=pathlib.Path(__file__).parent;v=json.load(open(W/'trace-values.json'))
def un(x):
 if isinstance(x,dict):
  if 'type'in x and 'contents'in x:return un(x['contents'])
  return {k:un(y) for k,y in x.items()}
 if isinstance(x,list):return [un(y) for y in x]
 return x
def flat(x):
 if isinstance(x,dict):return sum((flat(a) for a in x.values()),[])
 if isinstance(x,list):return sum((flat(a) for a in x),[])
 return [x]
def shape(x):
 if not isinstance(x,list):return []
 if not x:return [0]
 sh=[shape(a) for a in x];assert all(a==sh[0] for a in sh)
 return [len(x)]+sh[0]
checks=collections.Counter()
for x in v:
 a=un(x['value']);nums=[a for a in flat(a) if isinstance(a,(float,int))];assert all(math.isfinite(n) for n in nums);checks['finite_compound_values']+=1
 if x['shape'] is not None:assert shape(a)==x['shape'];checks['tensor_shapes']+=1
 if x['variable']=='times':assert all(n>=0 for n in nums);checks['nonnegative_timing_snapshots']+=1
 if 'softmax' in x['function'] and x['variable'].startswith('y'):
  for row in a:assert abs(sum(row)-1)<1e-6 and all(0<=n<=1 for n in row)
  checks['softmax_arrays']+=1
 if 'row_sum' in x['function'] and x['variable'].startswith('y'):assert a==[10,26];checks['row_sum_arrays']+=1
 if x['variable']=='vocab':
  for i in range(256):assert ast.literal_eval(a[str(i)])==bytes([i])
  for i,t in [(256,b'th'),(257,b'the'),(258,b'the ')]:
   if str(i)in a:assert ast.literal_eval(a[str(i)])==t
  checks['bpe_vocab_snapshots']+=1
 if x['function']=='locality_sensitive_hashing' and x['variable']=='probs':
  found=[(b,r) for b in (5,10,20,25,50) for r in (5,10,20) if all(abs(p-(1-(1-float(s)**r)**b))<1e-12 for s,p in a.items())]
  assert found;checks['lsh_probability_tables']+=1
for i in (256,):assert abs(sum(un(v[i]['value']))-1)<1e-10
assert all(abs(b-a/sum(v[255]['value']))<1e-12 for a,b in zip(v[255]['value'],v[256]['value']))
for i,batch in [(393,1),(394,64),(395,256),(396,64),(397,64),(398,256)]:
 a=un(v[i]['value']);assert math.isclose(a['latency'],a['memory']/3.35e12);assert math.isclose(a['throughput'],batch/a['latency']);checks['symbolic_stats']+=1
assert math.isclose(v[278]['value'][0][0],math.sqrt(.8/4),rel_tol=1e-7)
for i in (279,280):assert all(abs(x-y/(math.sqrt(.8/4)+1e-5))<1e-6 for x,y in zip(v[i]['value'][0],v[276]['value'][0]))
result={'checks':dict(checks),'scope':'Full finite-value and recorded tensor-shape inventory; selected algebraic relationships recomputed. Random model outputs and GPU benchmark timings are not independently reproduced.'}
(W/'trace-value-checks.json').write_text(json.dumps(result,indent=2));print(result)
