# Semantic segmentation of historical maps

**Paper**: Urbanisation and urban divergence: France 1760 – 2020

**Authors**

- [Clément Gorin](https://www.clementgorin.com/), Pantheon-Sorbonne University
- [Pierre-Philippe Combes](https://sites.google.com/view/pierrephilippecombes/), Sciences Po Paris & CNRS
- [Gilles Duranton](https://real-faculty.wharton.upenn.edu/duranton/), University of Pennsylvania
- [Laurent Gobillon](http://laurent.gobillon.free.fr/), Paris School of Economics
- [Frédéric Robert-Nicoud](https://frobertnicoud.weebly.com/), University of Geneva

**Corresponding author**: Clément Gorin, clement.gorin@univ-paris1.fr

## Data

The *SCAN50 historique* collection of maps covers mainland France and Corse at the end of the 1950's. The database consists of 1023 raster tiles covering an area of 25 km2 with a 5 x 5 m resolution. The geocoded rasters can be downloaded form the [French National Geographical Institute (IGN)](https://geoservices.ign.fr/scanhisto).

Key | Value
--- | ---
Number rasters | 1023
Extent | 5000 x 5000
Resolution | 5 x 5 metres
CRS | Lambert 93 (EPSG:2154)

This collection is a patchwork of five different map types, with a varying number of representations and colours. 

<img src='https://www.dropbox.com/scl/fi/5bmj6eykyu4dk7tzq3yqx/fig1960_types.jpg?rlkey=ykazxnli19s3447tnn4nu2mrd&dl=1'>

The annotated dataset was produced by manually vectorising 17 map tiles. We are grateful to Olena Bogdan, Célian Jounin, Siméon Mangematin, Matéo Moglia, Yoann Mollier-Loison, Rémi Pierotti, and Nathan Vieira for their invaluable research assistance. The training sample includes both urban and non-urban areas and is approximately balanced across the various legend categories.

## Model

<img src='https://www.dropbox.com/scl/fi/wgvqiix6scbm1c983ephk/segmentation_model.jpg?rlkey=71blfq7cm1yzw26puu4l0ybhs&dl=1'>