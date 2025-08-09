# TorchSharp vs pytorch

Testing ground to compare pytorch + TorchSharp

# Quick start
```sh
cd pytorch
uv sync
uv run simple.py
uv run chess_rl_sim.py
uv run python -m cProfile chess_rl_sim.py > profile.txt
cd..
cd csharp
dotnet run simple
dotnet run chess
dotnet-trace collect -o chess_cpu.nettrace -- ./bin/Debug/net8.0/csharp chess cpu
dotnet trace report chess_cpu.nettrace topN -n 20 > profile_cpu.txt
```

# todo
- try to make C# chess sim as fast as py
    - cpu is close enough
        py Done training in 3.74s. 26.72 eps/sec, 1362.62 moves/sec
        C# Done training in 4.29s. 23.29 eps/sec, 1187.64 moves/sec
    - gpu is ~80% py speed
        py Done training in 2.57s. 38.85 eps/sec, 1981.55 moves/sec
        C# Done training in 3.14s. 31.86 eps/sec, 1624.79 moves/sec



dotnet cpu:
Top 20 Functions (Exclusive)                                                  Inclusive           Exclusive
1.  torch+Tensor.Finalize()                                                   45.43%              40.53%
2.  torch+nn+functional.conv2d(class Tensor,class Tensor,class Tensor,int6    21.86%              21.86%
3.  torch._tensor_generic(class System.Array,value class System.ReadOnlySp    7.55%               6.73%
4.  torch+Tensor.Dispose(bool)                                                5.03%               5.03%
5.  torch+nn+functional.linear(class Tensor,class Tensor,class Tensor)        4.54%               4.54%
6.  torch+Tensor.size()                                                       4.53%               4.5%
dotnet gpu
Top 20 Functions (Exclusive)                                                  Inclusive           Exclusive
1.  torch+Tensor.Finalize()                                                   40.68%              34.19%
2.  torch+nn+functional.conv2d(class Tensor,class Tensor,class Tensor,int6    10.67%              10.67%
3.  torch._tensor_generic(class System.Array,value class System.ReadOnlySp    9.04%               7.83%
4.  torch+Tensor.Dispose(bool)                                                6.93%               6.92%
5.  torch+Tensor.size()                                                       5.56%               5.53%
py cpu
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     5100    0.121    0.000    4.082    0.001 chess_rl_sim.py:147(get_action)
     5100    0.090    0.000    2.990    0.001 chess_rl_sim.py:96(forward)
    15300    0.015    0.000    1.907    0.000 conv.py:553(forward)
    15300    0.012    0.000    1.884    0.000 conv.py:536(_conv_forward)
py gpu
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     5100    0.125    0.000    2.930    0.001 chess_rl_sim.py:149(get_action)
    15300    0.014    0.000    0.604    0.000 conv.py:553(forward)
    15300    0.011    0.000    0.583    0.000 conv.py:536(_conv_forward)













dotnet cpu:
Top 20 Functions (Exclusive)                                                  Inclusive           Exclusive
1.  torch+Tensor.Finalize()                                                   45.43%              40.53%
2.  torch+nn+functional.conv2d(class Tensor,class Tensor,class Tensor,int6    21.86%              21.86%
3.  torch._tensor_generic(class System.Array,value class System.ReadOnlySp    7.55%               6.73%
4.  torch+Tensor.Dispose(bool)                                                5.03%               5.03%
5.  torch+nn+functional.linear(class Tensor,class Tensor,class Tensor)        4.54%               4.54%
6.  torch+Tensor.size()                                                       4.53%               4.5%
7.  NativeLibrary.LoadLibraryByName(class System.String,class System.Refle    3.72%               3.72%
8.  FakeChessGame.RandomState()                                               3.61%               2.99%
9.  torch+Tensor.relu()                                                       2.14%               2.14%
10. SafeHandle.Finalize()                                                     1.29%               1.29%
11. torch+nn+functional.dropout(class Tensor,float64,bool,bool)               1.13%               1.13%
12. Buffer._Memmove(unsigned int8&,unsigned int8&,unsigned int)               0.77%               0.77%
13. torch.stack(class System.Collections.Generic.IEnumerable`1<class Tenso    0.63%               0.63%
14. Random.NextDouble()                                                       0.62%               0.62%
dotnet gpu
Top 20 Functions (Exclusive)                                                  Inclusive           Exclusive
1.  torch+Tensor.Finalize()                                                   40.68%              34.19%
2.  torch+nn+functional.conv2d(class Tensor,class Tensor,class Tensor,int6    10.67%              10.67%
3.  torch._tensor_generic(class System.Array,value class System.ReadOnlySp    9.04%               7.83%
4.  torch+Tensor.Dispose(bool)                                                6.93%               6.92%
5.  torch+Tensor.size()                                                       5.56%               5.53%
6.  NativeLibrary.LoadLibraryByName(class System.String,class System.Refle    4.89%               4.89%
7.  torch+nn+functional.linear(class Tensor,class Tensor,class Tensor)        4.45%               4.45%
8.  FakeChessGame.RandomState()                                               4.67%               3.65%
9.  torch+Tensor.relu()                                                       2.98%               2.98%
10. torch+Tensor.to(value class TorchSharp.DeviceType,int32,bool,bool,bool    2.78%               2.72%
11. torch+Tensor.ToScalar()                                                   2.33%               2.33%
12. Scalar.Dispose(bool)                                                      1.44%               1.44%
13. torch+nn+functional.dropout(class Tensor,float64,bool,bool)               1.36%               1.36%
14. torch+Tensor.to(value class ScalarType,class Device,bool,bool,bool)       1.21%               1.21%
15. Buffer._Memmove(unsigned int8&,unsigned int8&,unsigned int)               1.18%               1.18%
16. Random.NextDouble()                                                       1.02%               1.02%
py cpu
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    4.198    4.198 chess_rl_sim.py:1(<module>)
        1    0.000    0.000    4.198    4.198 chess_rl_sim.py:21(main)
      2/1    0.024    0.012    4.198    4.198 chess_rl_sim.py:184(train_against)
     5100    0.121    0.000    4.082    0.001 chess_rl_sim.py:147(get_action)
      284    0.006    0.000    3.953    0.014 __init__.py:1(<module>)
40800/5100    0.044    0.000    3.007    0.001 module.py:1747(_wrapped_call_impl)
40800/5100    0.051    0.000    3.001    0.001 module.py:1755(_call_impl)
     5100    0.090    0.000    2.990    0.001 chess_rl_sim.py:96(forward)
    15300    0.015    0.000    1.907    0.000 conv.py:553(forward)
    15300    0.012    0.000    1.884    0.000 conv.py:536(_conv_forward)
    15300    1.872    0.000    1.872    0.000 {built-in method torch.conv2d}
       12    0.000    0.000    1.128    0.094 _jit_internal.py:1022(_overload)
   212/83    0.001    0.000    1.120    0.013 _ops.py:316(fallthrough)
        1    0.000    0.000    0.638    0.638 triton_kernel_wrap.py:797(__init__)
       29    0.001    0.000    0.548    0.019 utils.py:1(<module>)
    58939    0.309    0.000    0.490    0.000 chess_rl_sim.py:50(step)
2172/2164    0.003    0.000    0.442    0.000 <frozen importlib._bootstrap>:806(module_from_spec)
        2    0.000    0.000    0.433    0.216 exc.py:1(<module>)
    27/26    0.000    0.000    0.407    0.016 <frozen importlib._bootstrap_external>:1287(create_module)
py gpu
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      284    0.006    0.000    3.949    0.014 __init__.py:1(<module>)
        1    0.000    0.000    3.046    3.046 chess_rl_sim.py:1(<module>)
        1    0.000    0.000    3.046    3.046 chess_rl_sim.py:21(main)
      2/1    0.029    0.014    3.046    3.046 chess_rl_sim.py:186(train_against)
     5100    0.125    0.000    2.930    0.001 chess_rl_sim.py:149(get_action)
40800/5100    0.041    0.000    1.618    0.000 module.py:1747(_wrapped_call_impl)
40800/5100    0.052    0.000    1.612    0.000 module.py:1755(_call_impl)
     5100    0.103    0.000    1.599    0.000 chess_rl_sim.py:98(forward)
       12    0.000    0.000    1.228    0.102 _jit_internal.py:1022(_overload)
   210/83    0.001    0.000    1.114    0.013 _ops.py:316(fallthrough)
        1    0.000    0.000    0.636    0.636 triton_kernel_wrap.py:797(__init__)
    15300    0.014    0.000    0.604    0.000 conv.py:553(forward)
    15300    0.011    0.000    0.583    0.000 conv.py:536(_conv_forward)
    15300    0.572    0.000    0.572    0.000 {built-in method torch.conv2d}
       29    0.001    0.000    0.548    0.019 utils.py:1(<module>)
    58123    0.299    0.000    0.473    0.000 chess_rl_sim.py:52(step)
        2    0.000    0.000    0.432    0.216 exc.py:1(<module>)
2172/2164    0.003    0.000    0.428    0.000 <frozen importlib._bootstrap>:806(module_from_spec)
        7    0.000    0.000    0.397    0.057 inductor_prims.py:18(make_prim)
    27/26    0.000    0.000    0.393    0.015 <frozen importlib._bootstrap_external>:1287(create_module)
