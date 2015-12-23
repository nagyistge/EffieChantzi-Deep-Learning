## Exercise 6

It contains all the .m, .mat, .avi files needed for tasks A and B, as well as all the obtained results. <br />

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

The following steps are required in order to create the datasets, which are saved as .mat files under /data.

* Image dataset from time-lapse microscopy movies

> Training Dataset

movieObjArray = createMovieObjects('movies');<br/>
[images_40, movieAxis, mean_intensity] = datasetFromAllMovies(movieObjArray, 93, 40, 40, 10);<br/>
[r, c] = find(mean_intensity < 0.009);<br/>
images_40 = images_40(:, ~ismember(1:size(images_40, 2), c));<br/>

> Test Dataset

[test_images_40, movieAxis, mean_intensity] = datasetFromAllMovies(movieObjArray, 93, 40, 40, 25);<br/>
[r, c] = find(mean_intensity < 0.02);<br/>
test_images_40 = test_images_40(:, ~ismember(1:size(test_images_40, 2), c));<br/>

<br/>
* mRNA gene expression dataset

glio_mRNA_data = GEOSeriesData('GSE23806');
