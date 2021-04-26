import tensorflow as tf

def seq2seq_window_dataset(series,
                           window_size,
                           batch_size=32,
                           shuffle_buffer=1000):
  

    series = tf.expand_dims(series, axis=-1)

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)

    ds = ds.flat_map(lambda w: w.batch(window_size + 1))

    ds = ds.shuffle(shuffle_buffer)

    ds = ds.map(lambda w: (w[:-1], w[1:]))

    return ds.batch(batch_size).prefetch(1)

# ---------------------------------------------------------------------------------------------------------------- 

def model_forecast(model, series, window_size):
  
    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size, shift=1, drop_remainder=True)
    
    ds = ds.flat_map(lambda w: w.batch(window_size))

    ds = ds.batch(32).prefetch(1)

    forecast = model.predict(ds)
    
    return forecast