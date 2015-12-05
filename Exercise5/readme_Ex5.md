## Exercise 5

It contains all the .m, .mat and .avi files needed for tasks A, B and C. <br />

### `Task A`

* PCA Compression

> Run the following in the command prompt and a menu will guide you through

```
compressionPCA
```

* Ordinary Stacked Autoencoder Initialization

> Run the following in the command prompt and a menu will guide you through

```
SAE
```

* Denoising Stacked Autoencoder Initialization

> Run the following in the command prompt and a menu will guide you through

```
DSAE
```


### `Task B`

> Run the following in the command prompt and a menu will guide you through

``` 
taskB
```


### `Task C`

* PCA Initialization

> Run the following in the command prompt and a menu will guide you through

``` 
PCA
```


* Ordinary Stacked Autoencoder Initialization

> Run the following in the command prompt and a menu will guide you through

``` 
SAE
```
<br /><br />

### `Datasets Creation`

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



