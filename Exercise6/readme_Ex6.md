## Exercise 6

It contains all the .m, .mat, .avi files needed for tasks A and B, as well as all the obtained results under /results. <br />

### `Task A`

* Prediction Modeling

> Run the following in the command prompt and a menu will guide you through

```
CV
```


### `Task B`

* Finding non-linear relationships by using deep learning

> Run the following in the command prompt and a menu will guide you through

``` 
CV_B
```

<br /><br />

### `Datasets Creation For Task A`

The following steps are required in order to create the image datasets, which are saved as .mat files under /data.

movieObjArray = createMovieObjects('movies');<br/>

* Movie 18 - Large interphase nuclei

> Training Dataset

 [images_m18_1, movieAxis_m18_1, mean_intensity_m18_1] = datasetFromAllMovies(movieObjArray, 93, 45, 45, 15);<br/>
 [r_m18_1, c_m18_1] = find(mean_intensity_m18_1 < 0.009);<br/>
 [images_m18_2, movieAxis_m18_2, mean_intensity_m18_2] = datasetFromAllMovies(movieObjArray, 93, 45, 45, 0);<br/>
 [r_m18_2, c_m18_2] = find(mean_intensity_m18_2 < 0.009);<br/>
 [images01, predictions01] = predictionDataset(images_m18_1, movieAxis_m18_1(3), 93);<br/>
 [images02, predictions02] = predictionDataset(images_m18_2, movieAxis_m18_2(3), 93);<br/>
 train_images_45_m18(:, 1 : 92) = images01;<br/>
 train_images_45_m18(:, 93 : 184) = images02;<br/>
 train_predictions_45_m18(:, 1 : 92) = predictions01;<br/>
 train_predictions_45_m18(:, 93 : 184) = predictions02;<br/>

> Test Dataset

 [images_m18_t, movieAxis_m18_t, mean_intensity_m18_t] = datasetFromAllMovies(movieObjArray, 93, 45, 45, -1);<br/>
 [r_m18_t, c_m18_t] = find(mean_intensity_m18_t < 0.009);<br/>
 [test_images_45_m18, test_predictions_45_m18] = predictionDataset(images_m18_t, movieAxis_m18_t(3), 93);<br/>
 
 
 * Movie 28 - Fragmented nuclei

> Training Dataset

 [images_m28_1, movieAxis_m28_1, mean_intensity_m28_1] = datasetFromAllMovies(movieObjArray, 93, 45, 45, 15);<br/>
 [r_m28_1, c_m28_1] = find(mean_intensity_m28_1 < 0.009);<br/>
 [images_m28_2, movieAxis_m28_2, mean_intensity_m28_2] = datasetFromAllMovies(movieObjArray, 93, 45, 45, 0);<br/>
 [r_m28_2, c_m28_2] = find(mean_intensity_m28_2 < 0.009);<br/>
 [images01, predictions01] = predictionDataset(images_m28_1, movieAxis_m28_1(13), 93);<br/>
 [images02, predictions02] = predictionDataset(images_m28_2, movieAxis_m28_2(13), 93);<br/>
 train_images_45_m28(:, 1 : 92) = images01;<br/>
 train_images_45_m28(:, 93 : 184) = images02;<br/>
 train_predictions_45_m28(:, 1 : 92) = predictions01;<br/>
 train_predictions_45_m28(:, 93 : 184) = predictions02;<br/>

> Test Dataset

 [images_m28_t, movieAxis_m28_t, mean_intensity_m28_t] = datasetFromAllMovies(movieObjArray, 93, 45, 45, -1);<br/>
 [r_m28_t, c_m28_t] = find(mean_intensity_m28_t < 0.009);<br/>
 [test_images_45_m28, test_predictions_45_m28] = predictionDataset(images_m28_t, movieAxis_m28_t(13), 93);<br/>