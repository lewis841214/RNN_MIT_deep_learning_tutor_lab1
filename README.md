
# RNN_MIT_deep_learning_tutor_lab1
一些註解

```
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

'''TODO: use the batch function to generate sequences of the desired size.'''
'''Hint: youll want to set drop_remainder=True'''
sequences = char_dataset.batch(seq_length+1,drop_remainder=True)
```
tf.data.Dataset.from_tensor_slices 這個指令是initilize tf.data.Dataset這個class的方法。它返回了一個dataset的class，裡面有iter這個東西讓你可以用for迴圈去iter。
原本text_as_int是一個np.array，然後進來之後它就變成一個data_adapter，一次iter_next可以按照順序一個一個呼叫這個array裡面的東西。
那
char_dataset.batch這個東西跟之後的函數map有點像，它把iter吃的東西改成seq_length+1的長度，也就是說原本一次吃一個，現在一次吃seq_length+1 length的長度的東西，drop_remainder=True則是，如果最後長度<seq_length+1時，則選擇被這個data放棄掉。

```
def split_input_target(chunk):
    input_text =  chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
    
dataset = sequences.map(lambda x: split_input_target(x))
```
dataset.sequences.map
給定一個function，這個function會去properly把你的dataset切片。
例如說，現在sequences是一個list的狀態(sequence現在是一個data_adapter啦，不過吐出來的東西是一個list) 那lambda x: 就是指這個list，然後把它套入這個function會被切成兩個list，長度各為len(x)-1
不過到目前為止，都還是在處理這個data_adapter吐出來的東西應該長什麼樣子。這是一連串的mapping。

```
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
```
這邊有提到buffer_size在 dataset.shuffle中的作用
https://juejin.im/post/5b855d016fb9a01a1a27d035
![image]https://github.com/lewis841214/RNN_MIT_deep_learning_tutor_lab1/blob/master/dataset_shuffle_buffer_size.png
或是stack_overflow(https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle)裡面的解答
By contrast, the buffer_size argument to tf.data.Dataset.shuffle() affects the randomness of the transformation. We designed the Dataset.shuffle() transformation (like the tf.train.shuffle_batch() function that it replaces) to handle datasets that are too large to fit in memory. Instead of shuffling the entire dataset, it maintains a buffer of buffer_size elements, and randomly selects the next element from that buffer (replacing it with the next input element, if one is available). Changing the value of buffer_size affects how uniform the shuffling is: if buffer_size is greater than the number of elements in the dataset, you get a uniform shuffle; if it is 1 then you get no shuffling at all. For very large datasets, a typical "good enough" approach is to randomly shard the data into multiple files once before training, then shuffle the filenames uniformly, and then use a smaller shuffle buffer. However, the appropriate choice will depend on the exact nature of your training job.
為什麼tensorflow的buffer這樣設計?因為他的data_adapter吃資料的方式就是，一次吃一定量的資料，所以他shuffle的方式並不能一次輸入整個dataset然後對datasetshuffle，而是一次吃一個buffer_size，然後從中隨機輸出

當初我會認為說，這個tutor中使用buffer就是一個把妳data用壞的方式。所以當初一直看不懂shuffle的原理到底是什麼。
但其實我搞錯了，因為現在這個dataset吐出來的東西，一次是(batch_size,sequence_length)的大小，也就是說，他是吃了buffer_size大小的(batch_size,sequence_length)這個東西，例如吃了1000個這個，然後再隨機輸出一個，並不會打壞我們的data，這樣其實蠻好的!








```
function test() {
  console.log("notice the blank line before this function?");
}
# Length of the vocabulary in chars
vocab_size = len(vocab)
print('vocalb_size=',vocab_size)
# The embedding dimension 
embedding_dim = 84

# The number of RNN units
rnn_units = 1024
```
這邊的rnn_units是ht的大小，我們從
![image](https://github.com/lewis841214/RNN_MIT_deep_learning_tutor_lab1/blob/master/ht.png)
可以看出，ht[i]屬於-1 至 1之間，也就是說若把-1 跟1 算成兩個資訊，其可以儲存之資訊量為2^1024

