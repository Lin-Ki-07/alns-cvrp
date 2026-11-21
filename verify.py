"""执行与解结构及搜索行为相关的基本验证。"""
import math
import random
import main as m

def verify():
    initial = m.build_initial_solution()
    assert m.is_feasible(initial)
    for destroy in m.DESTROY_OPERATORS:
        for count in [1,4,len(m.CUSTOMERS)]:
            original = m.copy_solution(initial)
            partial,removed = destroy(initial,count,random.Random(42))
            assert initial == original
            assert len(removed)==len(set(removed))==count
            assert sorted([i for r in partial for i in r]+removed)==m.CUSTOMERS
            for repair in m.REPAIR_OPERATORS:
                backup = m.copy_solution(partial)
                candidate = repair(partial,removed,random.Random(42))
                assert partial == backup
                assert candidate is None or m.is_feasible(candidate)
    for r in initial:
        for p,i in enumerate(r):
            shortened = r[:p]+r[p+1:]
            assert math.isclose(m.route_distance(r)-m.route_distance(shortened),m.removal_saving(r,p),abs_tol=1e-9)
    a = m.run_alns()
    b = m.run_alns()
    assert a == b, '固定种子不可复现'
    for seed in range(10):
        result = m.run_alns(iterations=300,seed=seed)
        assert m.is_feasible(result['best_solution'])
        assert result['best_cost']<=result['initial_cost']+1e-9
        h = result['history']
        assert len(h['iteration'])==301
        assert all(y<=x+1e-9 for x,y in zip(h['best_cost'],h['best_cost'][1:]))
        assert all(best<=current+1e-9 for best,current in zip(h['best_cost'],h['current_cost']))
    print('PASS: 初始解、两类破坏/修复、客户覆盖、容量、移除距离增量、输入隔离。')
    print('PASS: 默认2000轮固定种子复现；10个种子各300轮的可行性和历史最优单调性。')

if __name__=='__main__':
    verify()
