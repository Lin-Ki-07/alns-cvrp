"""博客配套的基础 ALNS/CVRP 示例。运行 python main.py 生成结果和图片。"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

COORDS = [(50,50),(20,60),(18,54),(25,48),(30,70),(38,75),
          (65,70),(72,62),(68,48),(60,30),(48,22),(35,28),(25,35)]
DEMANDS = [0,4,3,5,2,4,3,5,2,4,3,4,3]
CAPACITY = 15
MAX_VEHICLES = 4
CUSTOMERS = list(range(1, len(COORDS)))
DIST = [[math.hypot(a-c,b-d) for c,d in COORDS] for a,b in COORDS]

def route_load(route):
    return sum(DEMANDS[i] for i in route)

def route_distance(route):
    nodes = [0] + route + [0]
    return sum(DIST[a][b] for a,b in zip(nodes,nodes[1:]))

def objective(solution):
    return sum(map(route_distance, solution))

def copy_solution(solution):
    return [r.copy() for r in solution]

def is_feasible(solution):
    return (1 <= len(solution) <= MAX_VEHICLES
            and all(solution)
            and sorted(i for r in solution for i in r) == CUSTOMERS
            and all(route_load(r) <= CAPACITY for r in solution))

def insertion_cost(route, customer, position):
    previous = 0 if position == 0 else route[position-1]
    following = 0 if position == len(route) else route[position]
    return DIST[previous][customer]+DIST[customer][following]-DIST[previous][following]

def build_initial_solution():
    if any(not 0 < DEMANDS[i] <= CAPACITY for i in CUSTOMERS):
        raise ValueError('客户需求量必须为正且不超过容量。')
    solution = []
    for customer in sorted(CUSTOMERS, key=lambda i:(-DEMANDS[i],i)):
        options = [(insertion_cost(r,customer,p),k,p)
                   for k,r in enumerate(solution)
                   if route_load(r)+DEMANDS[customer] <= CAPACITY
                   for p in range(len(r)+1)]
        if options:
            _,k,p = min(options)
            solution[k].insert(p,customer)
        elif len(solution) < MAX_VEHICLES:
            solution.append([customer])
        else:
            raise RuntimeError('贪心初始化失败，不代表问题无解。')
    assert is_feasible(solution)
    return solution

def random_removal(solution, remove_count, rng):
    customers = [i for r in solution for i in r]
    if not 1 <= remove_count <= len(customers):
        raise ValueError('非法移除数量。')
    removed = rng.sample(customers,remove_count)
    removed_set = set(removed)
    partial = [[i for i in r if i not in removed_set] for r in solution]
    return [r for r in partial if r], removed

def removal_saving(route, position):
    i = route[position]
    u = 0 if position == 0 else route[position-1]
    v = 0 if position == len(route)-1 else route[position+1]
    return DIST[u][i]+DIST[i][v]-DIST[u][v]

def worst_removal(solution, remove_count, rng, randomness=3.0):
    if not 1 <= remove_count <= sum(map(len,solution)) or randomness < 1:
        raise ValueError('非法移除参数。')
    partial,removed = copy_solution(solution),[]
    for _ in range(remove_count):
        options = [(removal_saving(r,p),k,p,i)
                   for k,r in enumerate(partial) for p,i in enumerate(r)]
        options.sort(key=lambda item:item[0],reverse=True)
        rank = int(rng.random()**randomness*len(options))
        _,k,p,i = options[rank]
        partial[k].pop(p)
        removed.append(i)
        if not partial[k]:
            partial.pop(k)
    return partial,removed

def insertion_options(solution, customer):
    options = []
    for k,r in enumerate(solution):
        if route_load(r)+DEMANDS[customer] > CAPACITY:
            continue
        p = min(range(len(r)+1),key=lambda p:insertion_cost(r,customer,p))
        options.append((insertion_cost(r,customer,p),k,p))
    if len(solution)<MAX_VEHICLES and DEMANDS[customer]<=CAPACITY:
        options.append((DIST[0][customer]+DIST[customer][0],len(solution),0))
    return sorted(options,key=lambda item:item[0])

def apply_insertion(solution, customer, option):
    _,k,p = option
    if k == len(solution):
        solution.append([customer])
    else:
        solution[k].insert(p,customer)

def greedy_insertion(partial_solution, removed, rng):
    solution,pending = copy_solution(partial_solution),list(removed)
    while pending:
        choices = []
        for i in pending:
            options = insertion_options(solution,i)
            if options:
                choices.append((options[0][0],i,options[0]))
        if not choices:
            return None
        _,i,option = min(choices,key=lambda item:item[0])
        apply_insertion(solution,i,option)
        pending.remove(i)
    return solution if is_feasible(solution) else None

def regret2_insertion(partial_solution, removed, rng):
    solution,pending = copy_solution(partial_solution),list(removed)
    while pending:
        choices = []
        for i in pending:
            options = insertion_options(solution,i)
            if not options:
                return None
            regret = math.inf if len(options)==1 else options[1][0]-options[0][0]
            choices.append(((regret,-options[0][0]),i,options[0]))
        _,i,option = max(choices,key=lambda item:item[0])
        apply_insertion(solution,i,option)
        pending.remove(i)
    return solution if is_feasible(solution) else None

DESTROY_OPERATORS = [random_removal,worst_removal]
REPAIR_OPERATORS = [greedy_insertion,regret2_insertion]
SCORE_GLOBAL_BEST,SCORE_IMPROVEMENT,SCORE_ACCEPTED,SCORE_REJECTED = 8.,4.,1.,0.

def accept_candidate(current_cost,candidate_cost,temperature,rng):
    delta = candidate_cost-current_cost
    return delta<=0 or (temperature>0 and rng.random()<math.exp(-delta/temperature))

def initial_temperature(initial_cost,worsening_ratio=.05,acceptance_probability=.5):
    if initial_cost<=0 or worsening_ratio<=0 or not 0<acceptance_probability<1:
        raise ValueError('非法初始温度参数。')
    return -initial_cost*worsening_ratio/math.log(acceptance_probability)

def cooling_schedule(start_temperature,iterations,end_ratio=.01):
    if start_temperature<=0 or iterations<=0 or not 0<end_ratio<1:
        raise ValueError('非法降温参数。')
    return start_temperature*end_ratio,end_ratio**(1/iterations)

class OperatorSelector:
    def __init__(self,operator_count,history_retention=.8,minimum_weight=.1):
        if operator_count<=0 or not 0<history_retention<1 or not 0<minimum_weight<=1:
            raise ValueError('非法算子选择参数。')
        self.history_retention,self.minimum_weight = history_retention,minimum_weight
        self.weights = [1.]*operator_count
        self.scores = [0.]*operator_count
        self.uses = [0]*operator_count

    def select(self,rng):
        return rng.choices(range(len(self.weights)),weights=self.weights,k=1)[0]

    def record(self,index,score):
        self.scores[index] += score
        self.uses[index] += 1

    def update_weights(self):
        rho = self.history_retention
        for i in range(len(self.weights)):
            if self.uses[i]:
                self.weights[i] = max(self.minimum_weight,rho*self.weights[i]+(1-rho)*self.scores[i]/self.uses[i])
        self.scores = [0.]*len(self.weights)
        self.uses = [0]*len(self.weights)

def operator_score(candidate_cost,previous_current_cost,previous_best_cost,accepted):
    if candidate_cost is None:
        return SCORE_REJECTED
    if candidate_cost < previous_best_cost:
        return SCORE_GLOBAL_BEST
    if candidate_cost < previous_current_cost:
        return SCORE_IMPROVEMENT
    return SCORE_ACCEPTED if accepted else SCORE_REJECTED

def run_alns(iterations=2000,seed=42,removal_ratio_min=.15,removal_ratio_max=.35,
             segment_length=100,history_retention=.8,minimum_weight=.1,
             worsening_ratio=.05,acceptance_probability=.5,end_temperature_ratio=.01):
    if iterations<=0 or segment_length<=0 or not 0<removal_ratio_min<=removal_ratio_max<=1:
        raise ValueError('非法搜索参数。')
    rng = random.Random(seed)
    initial = build_initial_solution()
    current,best = copy_solution(initial),copy_solution(initial)
    initial_cost = current_cost = best_cost = objective(initial)
    ds = OperatorSelector(len(DESTROY_OPERATORS),history_retention,minimum_weight)
    rs = OperatorSelector(len(REPAIR_OPERATORS),history_retention,minimum_weight)
    temperature = initial_temperature(initial_cost,worsening_ratio,acceptance_probability)
    end_temperature,cooling_rate = cooling_schedule(temperature,iterations,end_temperature_ratio)
    n = len(CUSTOMERS)
    low = max(1,math.floor(n*removal_ratio_min))
    high = min(n,max(low,math.floor(n*removal_ratio_max)))
    history = {k:[] for k in ['iteration','current_cost','best_cost','temperature','destroy_weights','repair_weights']}
    def snapshot(t):
        for key,value in [('iteration',t),('current_cost',current_cost),('best_cost',best_cost),
                          ('temperature',temperature),('destroy_weights',ds.weights.copy()),('repair_weights',rs.weights.copy())]:
            history[key].append(value)
    snapshot(0)
    accepted_count = failed_repairs = best_updates = 0
    for iteration in range(1,iterations+1):
        old_current,old_best = current_cost,best_cost
        di,ri = ds.select(rng),rs.select(rng)
        partial,removed = DESTROY_OPERATORS[di](current,rng.randint(low,high),rng)
        candidate = REPAIR_OPERATORS[ri](partial,removed,rng)
        candidate_cost,accepted = None,False
        if candidate is None:
            failed_repairs += 1
        else:
            if not is_feasible(candidate):
                raise RuntimeError('修复算子返回不可行解。')
            candidate_cost = objective(candidate)
            accepted = accept_candidate(old_current,candidate_cost,temperature,rng)
            if accepted:
                current,current_cost = copy_solution(candidate),candidate_cost
                accepted_count += 1
            if candidate_cost<old_best:
                best,best_cost = copy_solution(candidate),candidate_cost
                best_updates += 1
        score = operator_score(candidate_cost,old_current,old_best,accepted)
        ds.record(di,score)
        rs.record(ri,score)
        if iteration%segment_length==0 or iteration==iterations:
            ds.update_weights()
            rs.update_weights()
        temperature = max(end_temperature,temperature*cooling_rate)
        snapshot(iteration)
    return dict(initial_solution=initial,initial_cost=initial_cost,best_solution=best,best_cost=best_cost,
                history=history,destroy_weights=ds.weights.copy(),repair_weights=rs.weights.copy(),
                accepted_count=accepted_count,failed_repairs=failed_repairs,best_updates=best_updates)

def save_figures(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(output/name,dpi=200,bbox_inches='tight')
        plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,5),sharex=True,sharey=True)
    for ax,solution,title in zip(axes,[result['initial_solution'],result['best_solution']],['初始配送方案','ALNS搜索得到的最好方案']):
        for k,r in enumerate(solution):
            nodes = [0]+r+[0]
            ax.plot([COORDS[i][0] for i in nodes],[COORDS[i][1] for i in nodes],label=f'车辆{k+1}（负载{route_load(r)}）')
        ax.scatter([COORDS[i][0] for i in CUSTOMERS],[COORDS[i][1] for i in CUSTOMERS],c='white',edgecolors='black',zorder=3)
        for i in CUSTOMERS:
            ax.annotate(str(i),COORDS[i],xytext=(5,5),textcoords='offset points')
        ax.scatter(*COORDS[0],marker='*',s=220,c='red',label='仓库',zorder=4)
        ax.set(title=f'{title}\n总距离：{objective(solution):.2f}',xlabel='横坐标',ylabel='纵坐标')
        ax.set_aspect('equal',adjustable='box')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    save(fig,'routes.png')
    h = result['history']
    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(h['iteration'],h['current_cost'],label='当前解',alpha=.65,lw=1)
    ax.plot(h['iteration'],h['best_cost'],label='历史最优解',lw=2)
    ax.set(xlabel='迭代次数',ylabel='总行驶距离',title='ALNS目标函数变化')
    ax.grid(alpha=.3)
    ax.legend()
    save(fig,'convergence.png')
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    for ax,key,operators,title in [(axes[0],'destroy_weights',DESTROY_OPERATORS,'破坏算子权重变化'),(axes[1],'repair_weights',REPAIR_OPERATORS,'修复算子权重变化')]:
        for i,op in enumerate(operators):
            ax.step(h['iteration'],[w[i] for w in h[key]],where='post',label=op.__name__)
        ax.set(xlabel='迭代次数',ylabel='算子权重',title=title)
        ax.grid(alpha=.3)
        ax.legend()
    save(fig,'weights.png')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs')
    args = parser.parse_args()
    result = run_alns(iterations=args.iterations,seed=args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    payload = {'parameters':{'iterations':args.iterations,'seed':args.seed},**result}
    (args.output/'result.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    h = result['history']
    with (args.output/'history.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['iteration','current_cost','best_cost','temperature'])
        writer.writerows(zip(h['iteration'],h['current_cost'],h['best_cost'],h['temperature']))
    save_figures(result,args.output)
    improvement = (result['initial_cost']-result['best_cost'])/result['initial_cost']*100
    print(f"初始总距离：{result['initial_cost']:.2f}\n最好总距离：{result['best_cost']:.2f}\n相对初始解改善：{improvement:.2f}%")
    for k,r in enumerate(result['best_solution'],1):
        print(f'车辆{k}：{[0]+r+[0]}，负载{route_load(r)}，距离{route_distance(r):.2f}')
    print(f'输出目录：{args.output.resolve()}')

if __name__ == '__main__':
    main()
