#  PAG-Net: An Novel Image Dehazing Method using Patch-aware Guidance
## About Paper

We are delighted to inform everyone that our paper has been successfully accepted by IJCNN (2026)

## Abstract

Imagedehazingisafundamentaltaskinlow-level computervision,aimingtorestorehaze-freeimagesfromhazy observations.However,existingapproachesoftenfaceatrade-off betweenpreservingglobalstructuralconsistencyandrecovering fne-grainedlocaltextures.Toaddressthischallenge,wepropose apatch-awarestrategytoexplicitlyguidethedehazingprocess.Specifcally,weintroducethePatch-AwareModulationModule (PAMM),whichiscomposedofpatch-awareguidanceand sensitivity-guidedmodulation.First,thepatch-awareguidance generatesaspatiallyvaryingsensitivitymapthatcharacter-izestheglobaldistributionofhazedensityacrosstheimage.Subsequently,thissensitivitymapisemployedtomodulatethe globalcontextualfeaturesextractedbyadedicatedlarge-kernel convolution,therebyachievingamoreharmoniousintegration ofglobalstructureandlocaldetail.Extensiveexperimentson multiplebenchmarkdatasetsdemonstratethatourpatch-aware strategyconsistentlyimprovesbothquantitativemetricsand qualitativevisualfdelityoverstate-of-the-artmethods.

## Data

1. [RESIDE Dataset](https://sites.google.com/site/boyilics/website-builder/reside)


## Code

Partial implementation and the pre-train models is available in the [code](PAG-Net0330/code) folder.  

## Performance

1. RESIDE-IN

![](MFF-Net/picture/MMFlood_result.png)

2. RESIDE-OUT

![](MFF-Net/picture/sen1flood.jpg)

3. NH-HAZE

![](MFF-Net/picture/ETCI.jpg)
