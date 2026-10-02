```bash
> airun.exe --quantize models\Qwen2.5-0.5B-Instruct\model.safetensors models\Qwen2.5-0.5B-Instruct-int8\model.safetensors

[quantize] reading models\Qwen2.5-0.5B-Instruct\model.safetensors
[quantize] layer 4/24
[quantize] layer 8/24
[quantize] layer 12/24
[quantize] layer 16/24
[quantize] layer 20/24
[quantize] layer 24/24
[quantize] writing models\Qwen2.5-0.5B-Instruct-int8\model.safetensors...
[quantize] done in 17352.8 ms
```

```bash
> airun.exe models\Qwen2.5-0.5B-Instruct-int8\model.safetensors models\Qwen2.5-0.5B-Instruct-int8\vocab.json models\Qwen2.5-0.5B-Instruct-int8\merges.txt

[load] safetensors: 0.7 ms (459 tensors)
[tokenizer] vocab: 151646, merges: 151387
[load] tokenizer: 178.6 ms (vocab=151646)
[model] H=896 NH=14 NKV=2 NL=24 HD=64 KVD=128 INTER=4864 VOCAB=151936
[model] linear weights: pre-quantized INT8 (fast load)
[model] weights loaded
[load] model: 440.8 ms

  +----------------------------------------------------------------------------+
  |                                                                            |
  | Qwen2.5-0.5B-Instruct                                                      |
  | native C++ inference - SSE2 + threads                                      |
  |                                                                            |
  | type /help for commands, /exit to quit                                     |
  |                                                                            |
  +----------------------------------------------------------------------------+


  -- commands -------------------------------------------------------------
  |  /reset  /clear  /temp <n>  /help  /exit
  ----------------------------------------------------------------------------
  you > Hi
  aI  > Hello! How can I assist you today?
  prefill 34 tok | 1467 ms   |   gen 9 tok | 837 ms | 10.8 tok/s
  you > 用 Python 编写一个简单的快速排序算法实现。
  aI  > 下面是一个使用 Python 实现的简单快速排序算法示例：

``python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

# 示例使用
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = quick_sort(arr)
print("排序后的数组:", sorted_arr)
``

这个函数接受一个列表作为输入，返回一个按升序排列的新列表。快速排序算法的工作原理是选择一个“基准”元素（这里选择的是数组的中间元素），将所有比这个基准值大的元素移动到它的左边，剩下的都移到右边。然后重复这个过程直到所有元素都被排序。
  prefill 22 new (cached 43) | 953 ms   |   gen 213 tok | 18344 ms | 11.6 tok/s
  you > /exit

  bye.


```

Runs on Pentium G5420 without AVX2.
