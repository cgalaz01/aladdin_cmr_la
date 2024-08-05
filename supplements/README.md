# Supplements

## Data Example
| ![data](data/data_example.png "Cross section view and volume across the cardiac cycle") | 
|:--:|
| _<span id="SupFig1">SupFig. 1</span>_ The cross-sectional view of the left atrium (LA) (top) and the volume across the cardiac cycle (bottom) of a representative healthy case (26M), acquired using the proposed high-resolution 3D Cine MRI protocol. |

| ![data_h1](data/data_slice_1.gif "Slice view of volunteer 1 across the cardiac cycle") | 
|:--:|
| _<span id="SupVid1">SupVid. 1</span>_ Slice view of the left atrium of a volunteer (26M) acquired using the proposed high-resolution 3D Cine MRI protocol. |

| ![data_h2](data/data_slice_2.gif "Slice view of volunteer 2 across the cardiac cycle") | 
|:--:|
| _<span id="SupVid2">SupVid. 2</span>_ Slice view of the left atrium of a volunteer (24F) acquired using the proposed high-resolution 3D Cine MRI protocol. |


## Segmentation Results
| ![la_seg_healthy](segmentation/seg_healthy.gif "Left atrium segmentation results across the cardiac cycle") | 
|:--:|
| _<span id="SupVid3">SupVid. 3</span>_ The segmentation results across the cardiac cycle for a representative healthy case (26M) is shown. On the left, the estimated segmentation map in a single-slice view, and, on the right, the Dice score and Hausdorff distance.

| ![la_seg_patient](segmentation/seg_patient.gif "Left atrium segmentation results across the cardiac cycle") | 
|:--:|
| _<span id="SupVid4">SupVid. 4</span>_ The segmentation results across the cardiac cycle for a representative patient case (52F with myocarditis) is shown. On the left, the estimated segmentation map in a single-slice view, and, on the right, the Dice score and Hausdorff distance.



## Registration Results
| ![la_reg](registration/la_reg.gif "Left atrium registration results across the cardiac cycle") | 
|:--:|
| _<span id="SupVid5">SupVid. 5</span>_ The image registration results across the cardiac cycle for a representative case is shown. Each row displays a slice of the resulting registration for the models: ANTs, Vxm-seg, Vxm, and Aladdin-R, respectively. The plots illustrate the volume, Dice score, and Hausdorff distance achieved by each of the models. |

| ![dvf](registration/dvf_comparison_healthy.gif "Comparison of the estimated displacement vector fields of a healthy case") | 
|:--:|
| _<span id="SupVid6">SupVid. 6</span>_ Comparison of the estimated displacement vector fields achieved of a healthy case (26M), from left to right, by Aladdin-R, ANTs, Vxm, and Vxm-seg, respectively, across the cardiac cycle. |

| ![dvf](registration/dvf_comparison_patient.gif "Comparison of the estimated displacement vector fields of a patient case") | 
|:--:|
| _<span id="SupVid7">SupVid. 7</span>_ Comparison of the estimated displacement vector fields achieved of a patient case (52F with myocarditis), from left to right, by Aladdin-R, ANTs, Vxm, and Vxm-seg, respectively, across the cardiac cycle. |


## Global Results
| ![global_all](global_results/individual.png "The normalized volume, displacement magnitude, and strains across the cardiac cycle for each case") |
|:--:|
| _<span id="SupFig2">SupFig. 2</span>_ The normalized volume, displacement magnitude, and strains across the cardiac cycle, plotted separately for each case is shown. For each case, the following details are provided: age, sex, condition, minimum volume, maximum volume, pre-activation volume, ejection fraction (EF), and active ejection fraction (aEF). |


_<span id="SupTab1">SupTab. 1</span>_ Patient metrics including left atrial volume (LAV), left ventricular ejection fraction (LVEF), left atrial active ejection fraction (LAaEF), and left atrial ejection fraction (LAEF), as estimated from the 2-chamber (2CH), 4-chamber (4CH), and 3D views.
| Case | Sex | Age (years) | Condition                    | LAV (cm2) | LVEF (%) | LAaEF 2CH (%) | LAEF 2CH (%) | LAaEF 4CH (%) | LAEF 4CH (%) | LAaEF 2/4CH (%) | LAEF 2/4CH (%) | LAaEF 3D (%) | LAEF 3D (%) |
|------|-----|-------------|------------------------------|-----------|----------|---------------|--------------|---------------|--------------|-----------------|----------------|--------------|-------------|
| P1   | F   | 52          | Myocarditis                  | 21        | 47       | 45.2          | 55.6         | 32.5          | 52.8         | 39.6            | 54.8           | 23.4         | 43.7        |
| P2   | F   | 26          | History of syncope           | 18        | 58       | 40.2          | 67.7         | 39.0          | 65.8         | 44.5            | 67.6           | 26.5         | 56.2        |
| P3   | M   | 35          | History of syncope           | 25        | 58       | 44.1          | 59.2         | 26.1          | 61.9         | 33.2            | 58.5           | 20.6         | 46.9        |
| P4   | M   | 60          | Myocardial infraction        | 17        | 22       | 46.0          | 58.0         | 37.6          | 60.7         | 36.9            | 53.9           | 20.1         | 39.1        |
| P5   | F   | 57          | Non-Ischaemic cardiomyopathy | 19        | 51       | 39.3          | 56.9         | 37.7          | 53.7         | 47.8            | 57.5           | 30.1         | 44.5        |
| P6   | M   | 55          | Myocardial infraction        | 24        | 35       | 36.4          | 41.3         | 22.8          | 39.1         | 32.3            | 40.5           | 15.3         | 26.1        |
| P7   | F   | 66          | Hypertrophic cardiomyopathy  | 21        | 60       | 43.8          | 59.0         | 52.0          | 64.4         | 41.8            | 57.7           | 34.6         | 48.4        |
| P8   | F   | 79          | Myocardial infraction        | 23        | 62       | 45.9          | 55.3         | 39.8          | 65.3         | 37.6            | 57.6           | 35.3         | 48.0        |


| ![global_corr](global_results/correlation_matrix.png "Correlation matrix between estimated values from 2-Chamber, 4-Chamber and 3D Views") |
|:--:|
| _<span id="SupFig3">SupFig. 3</span>_ The correlation matrix between the left atrial volume (LAV), left ventricular ejection fraction (LVEF), left atrial active ejection fraction (LAaEF), and left atrial ejection fraction (LAEF), as estimated from the 2-chamber (2CH), 4-chamber (4CH), and 3D views. |


## Regional Results (Atlas-based)
| ![atlas](regional_results/atlas.gif "Overview of the atlas") | 
|:--:|
| _<span id="SupVid8">SupVid. 8</span>_ Overview of the constructed atlas from the available healthy population. The columns depict the whole heart view, the left atrium displacement vector field, and the first principal strain. The anterior and posterior views are shown in the first and second rows, respectively. |

| ![atlas_h1](regional_results/healthy1.gif "Analysis of a healthy case 1 using the atlas") | 
|:--:|
| _<span id="SupVid9">SupVid. 9</span>_ The Mahalanobis distance of a healthy case (26M) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| ![atlas_h2](regional_results/healthy2.gif "Analysis of a healthy case 2 using the atlas") | 
|:--:|
| _<span id="SupVid10">SupVid. 10</span>_ The Mahalanobis distance of a healthy case (22F) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| ![atlas_h3](regional_results/healthy3.gif "Analysis of a healthy case 3 using the atlas") | 
|:--:|
| _<span id="SupVid11">SupVid. 11</span>_ The Mahalanobis distance of a healthy case (24F) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| ![atlas_p1](regional_results/patient1.gif "Analysis of a patient case 1 using the atlas") | 
|:--:|
| _<span id="SupVid12">SupVid. 12</span>_ The Mahalanobis distance of a patient case (52F with myocarditis) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| ![atlas_p2](regional_results/patient2.gif "Analysis of a patient case 2 using the atlas") | 
|:--:|
| _<span id="SupVid13">SupVid. 13</span>_ The Mahalanobis distance of a patient case (60M with myocardial infarction) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| ![atlas_p3](regional_results/patient3.gif "Analysis of a patient case 3 using the atlas") | 
|:--:|
| _<span id="SupVid14">SupVid. 14</span>_ The Mahalanobis distance of a patient case (57F with non-ischaemic cardiomyopathy) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |
