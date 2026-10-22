"""CS336 original teaching reconstructions; Python standard library only.
Run: python3 CS336-教学小实验.py
Prints deterministic results. No GPU, network, model downloads, or file writes.
These are not reproductions of classroom GPU benchmarks.
"""
import collections,json,math

def bpe_toy():
    seq=list(b'banana banana bandana');initial=len(seq);merges=[];vocab={i:bytes([i]) for i in range(256)}
    for step in range(6):
        counts=collections.Counter(zip(seq,seq[1:]))
        pair=min(counts,key=lambda p:(-counts[p],p));token=256+step
        vocab[token]=vocab[pair[0]]+vocab[pair[1]]
        out=[];i=0
        while i<len(seq):
            if i+1<len(seq) and (seq[i],seq[i+1])==pair:out.append(token);i+=2
            else:out.append(seq[i]);i+=1
        seq=out;merges.append({'merge':vocab[token].decode(),'remaining_tokens':len(seq)})
    decoded=b''.join(vocab[x] for x in seq)
    assert decoded==b'banana banana bandana'
    return {'initial_byte_tokens':initial,'merges':merges,'final_tokens':len(seq),'round_trip':True,'limitation':'Toy corpus; merges across spaces; no production pretokenizer or special tokens.'}

def softmax(scores):
    top=max(scores);exps=[math.exp(x-top) for x in scores];total=sum(exps)
    return [x/total for x in exps]

def attention():
    q=k=[[1.,0.],[0.,1.]];values=[[10.,0.],[0.,10.]]
    def run(v,causal):
        result=[]
        for i,row in enumerate(q):
            scores=[sum(a*b for a,b in zip(row,col))/math.sqrt(2) if not causal or j<=i else -math.inf for j,col in enumerate(k)]
            weights=softmax(scores)
            result.append([sum(weights[j]*v[j][c] for j in range(2)) for c in range(2)])
        return result
    base=run(values,True);changed=run([[10.,0.],[999.,999.]],True)
    assert base[0]==changed[0]
    return {'causal_output':base,'future_value_changed_output':changed,'first_output_unchanged':True,'without_mask':run(values,False)}

def loss_and_stability():
    p=softmax([1000.,1001.]);assert abs(sum(p)-1)<1e-12
    try:math.exp(1000);overflow=False
    except OverflowError:overflow=True
    return {'stable_softmax_1000_1001':p,'naive_exp_1000_overflows':overflow,'loss_if_true_probability_0_8':-math.log(.8),'loss_if_true_probability_0_2':-math.log(.2)}

def accumulation():
    w=1.;lr=.1;data=[(1.,0.),(3.,2.)]
    def grad(w,x,y):return 2*(w*x-y)*x
    full=sum(grad(w,x,y) for x,y in data)/len(data)
    micro=0.
    for x,y in data:micro+=grad(w,x,y)/len(data)
    accumulated=w-lr*micro;full_w=w-lr*full
    sequential=w
    for x,y in data:sequential-=lr*grad(sequential,x,y)
    assert abs(accumulated-full_w)<1e-12 and abs(sequential-full_w)>1e-8
    return {'averaged_gradient':full,'full_batch_w':full_w,'accumulated_w':accumulated,'update_each_microbatch_w':sequential,'assumptions':'Same parameters during accumulation; mean loss; no dropout/batch-dependent layer; one optimizer step.'}

def collectives():
    inputs=[[float(c+r) for c in range(4)] for r in range(4)]
    summed=[sum(row[c] for row in inputs) for c in range(4)]
    shards=[[summed[r]] for r in range(4)];gathered=[x for shard in shards for x in shard]
    assert summed==[6.,10.,14.,18.] and summed==gathered
    return {'inputs':inputs,'all_reduce_sum':summed,'reduce_scatter_shards':shards,'then_all_gather':gathered,'limitation':'List arithmetic illustrates semantics; no actual distributed processes or performance measurement.'}

def lsh():
    b,r=3,4
    probs={str(s):1-(1-s**r)**b for s in [.2,.5,.8,.99]}
    assert 0<probs['0.2']<probs['0.8']<probs['0.99']<1
    return {'bands':b,'rows_per_band':r,'candidate_probability':probs,'assumptions':'Independent ideal MinHash values; candidate collision is not proof of duplicate identity.','mixture_example':{'total_tokens':1000,'source_tokens':10,'source_weight':.1,'expected_source_epochs':1000*.1/10}}

def policy_gradient():
    rewards=[0.,1.,1.,2.];mean=sum(rewards)/len(rewards);centered=[x-mean for x in rewards]
    w=2.;eps=1e-5
    sigmoid=lambda x:1/(1+math.exp(-x))
    frozen=sigmoid(w)
    fn=lambda x:sigmoid(x)/frozen
    wrong=lambda x:sigmoid(x)/sigmoid(x)
    d=(fn(w+eps)-fn(w-eps))/(2*eps);bad=(wrong(w+eps)-wrong(w-eps))/(2*eps)
    assert abs(d-(1-sigmoid(w)))<1e-9 and bad==0
    return {'rewards':rewards,'mean':mean,'centered_advantages':centered,'old_policy_detached_ratio_value':fn(w),'derivative_with_frozen_denominator':d,'derivative_if_both_depend_on_current_w':bad,'equal_rewards_center_to':[0.,0.,0.,0.]}

def recurrence():
    rows={}
    for a in [.5,-.5,-2.]:
        x=1.;values=[x]
        for _ in range(8):x=a*x;values.append(x)
        rows[str(a)]=values
    assert rows['-2.0'][-1]==256.
    return {'scalar_recurrences':rows,'point':'A negative coefficient alone does not guarantee stability; for x[t+1]=a*x[t], |a|<1 ensures convergence to 0.','matrix_counterexample':{'A':[[0,100],[0,0]],'spectral_radius':0,'spectral_2_norm':100,'A_squared':[[0,0],[0,0]]},'limitation':'Does not reproduce Parcae dynamics or prove stability of nonlinear Transformer loops.'}

def reward_counterexample():
    prompt=[1,1,2];correct=sorted(prompt);incorrect=[1,2,2]
    def score(response):return sum(x in response for x in prompt)+sum(x<=y for x,y in zip(response,response[1:]))
    a,b=score(correct),score(incorrect)
    assert correct!=incorrect and a==b==5
    return {'prompt':prompt,'correct':correct,'incorrect':incorrect,'reward_correct':a,'reward_incorrect':b,'point':'Membership without multiplicity plus adjacent ordering can give maximum reward to a wrong multiset.','provenance':'Original counterexample to the reward rule in 2025 lecture_17.py; not asserted to have been discussed by the lecturer.'}

def online_attention():
    # Original CPU reconstruction: a fixed query's scores and value vectors.
    scores = [1000., 1002., 999., 1005., 1001.]
    values = [[1., 0.], [0., 2.], [3., 1.], [-1., 4.], [2., -2.]]
    weights = [math.exp(x - max(scores)) for x in scores]
    dense = [sum(w * v[j] for w, v in zip(weights, values)) / sum(weights) for j in range(2)]
    outputs = {}
    for tile_size in [1, 2, 3, 5]:
        maximum, normalizer, numerator = -math.inf, 0., [0., 0.]
        for start in range(0, len(scores), tile_size):
            block = scores[start:start + tile_size]
            new_maximum = max(maximum, max(block))
            rescale = math.exp(maximum - new_maximum)
            block_weights = [math.exp(x - new_maximum) for x in block]
            normalizer = normalizer * rescale + sum(block_weights)
            numerator = [numerator[j] * rescale + sum(w * v[j] for w, v in zip(block_weights, values[start:start + tile_size])) for j in range(2)]
            maximum = new_maximum
        output = [v / normalizer for v in numerator]
        assert all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(output, dense))
        outputs[str(tile_size)] = output
    return {'dense': dense, 'tiled': outputs, 'point': 'Rescale both the running denominator and weighted-value numerator when a later tile raises the maximum.', 'limitation': 'One fixed query, precomputed scores, tiny CPU vectors; demonstrates exact algebra up to floating-point roundoff, not GPU FlashAttention performance.'}

def run_all():
    return {'D01_BPE':bpe_toy(),'D02_Attention':attention(),'D03_Stability_and_loss':loss_and_stability(),'D04_Gradient_accumulation':accumulation(),'D05_Collectives':collectives(),'D06_LSH_and_mixture':lsh(),'D07_Policy_gradient':policy_gradient(),'D08_Recurrence':recurrence(),'D09_Reward_counterexample':reward_counterexample(),'D10_Online_attention':online_attention()}

if __name__=='__main__':print(json.dumps(run_all(),ensure_ascii=False,indent=2))
