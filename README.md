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
[quantize] done in 16472.5 ms
```

```bash
> airun.exe models\Qwen2.5-0.5B-Instruct-int8\model.safetensors models\Qwen2.5-0.5B-Instruct-int8\vocab.json models\Qwen2.5-0.5B-Instruct-int8\merges.txt

[load] safetensors: 0.7 ms (458 tensors)
[tokenizer] vocab: 151646, merges: 151387
[load] tokenizer: 227.7 ms (vocab=151646)
[model] H=896 NH=14 NKV=2 NL=24 HD=64 KVD=128 INTER=4864 VOCAB=151936
[model] linear weights: pre-quantized INT8 (fast load)
[model] weights loaded
[load] model: 813.1 ms

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
  you > hi!
  aI  > Hello! How can I assist you today?
  prefill 35 tok | 1528 ms   |   gen 9 tok | 952 ms | 9.5 tok/s
  you > 用 Python 编写一个简单的快速排序算法实现。
  aI  > 当然，以下是一个使用 Python 实现的快速排序算法：

``python
def quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)

# Example usage:
if __name__ == "__main__":
    example_array = [3, 6, 8, 10, 1, 2, 1]
    print("Original array:", example_array)
    sorted_array = quicksort(example_array)
    print("Sorted array:", sorted_array)
``

这个代码定义了一个 `quicksort` 函数，它接受一个列表作为输入，返回一个排序后的列表。快速排序是一种在平均情况下效率较高的排序算法。
  prefill 22 new (cached 44) | 975 ms   |   gen 203 tok | 21339 ms | 9.5 tok/s
  you > /exit

  bye.


```

Runs on Pentium G5420 without AVX2.
