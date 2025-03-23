# Supplements

## Data Example
| <span id="SupFig1">Supplementary Figure 1</span> |
|:--:|
| ![data](data/data_example.png "Cross section view and volume across the cardiac cycle") | 
| _SupFig. 1_ The cross-sectional view of the left atrium (LA) (top) and the volume across the cardiac cycle (bottom) of a representative healthy case (26M), acquired using the proposed high-resolution 3D Cine MRI protocol. |

| <span id="SupVid1">Supplementary Video 1</span> |
|:--:|
| ![data_h1](data/data_slice_1.gif "Slice view of volunteer 1 across the cardiac cycle") | 
| _SupVid. 1_ Slice view of the left atrium of a volunteer (26M) acquired using the proposed high-resolution 3D Cine MRI protocol. |

| <span id="SupVid2">Supplementary Video 2</span> |
|:--:|
| ![data_h2](data/data_slice_2.gif "Slice view of volunteer 2 across the cardiac cycle") |
| _SupVid. 2_ Slice view of the left atrium of a volunteer (24F) acquired using the proposed high-resolution 3D Cine MRI protocol. |


## Segmentation Results
| <span id="SupVid3">Supplementary Video 3</span> |
|:--:|
| ![la_seg_healthy](segmentation/seg_healthy.gif "Left atrium segmentation results across the cardiac cycle") |
| _SupVid. 3_ The segmentation results across the cardiac cycle for a representative healthy case (26M) is shown. On the left, the estimated segmentation map in a single-slice view, and, on the right, the Dice score and Hausdorff distance.

| <span id="SupVid4">Supplementary Video 4</span> |
|:--:|
| ![la_seg_patient](segmentation/seg_patient.gif "Left atrium segmentation results across the cardiac cycle") | 
| _SupVid. 4_ The segmentation results across the cardiac cycle for a representative patient case (52F with myocarditis) is shown. On the left, the estimated segmentation map in a single-slice view, and, on the right, the Dice score and Hausdorff distance.



## Registration Results
| <span id="SupVid5">Supplementary Video 5</span> |
|:--:|
| ![la_reg](registration/la_reg.gif "Left atrium registration results across the cardiac cycle") | 
| _SupVid. 5_ The image registration results across the cardiac cycle for a representative case is shown. Each row displays a slice of the resulting registration for the models: ANTs, Vxm-seg, Vxm, and Aladdin-R, respectively. The plots illustrate the volume, Dice score, and Hausdorff distance achieved by each of the models. |

| <span id="SupVid6">Supplementary Video 6</span> |
|:--:|
| ![dvf](registration/dvf_comparison_healthy.gif "Comparison of the estimated displacement vector fields of a healthy case") | 
| _SupVid. 6_ Comparison of the estimated displacement vector fields achieved of a healthy case (26M), from left to right, by Aladdin-R, ANTs, Vxm, and Vxm-seg, respectively, across the cardiac cycle. |

| <span id="SupVid7">Supplementary Video 7</span> |
|:--:|
| ![dvf](registration/dvf_comparison_patient.gif "Comparison of the estimated displacement vector fields of a patient case") | 
| _SupVid. 7_ Comparison of the estimated displacement vector fields achieved of a patient case (52F with myocarditis), from left to right, by Aladdin-R, ANTs, Vxm, and Vxm-seg, respectively, across the cardiac cycle. |

| <span id="SupVid8">Supplementary Video 8</span> |
|:--:|
| ![dvf](registration/healthy1_results.gif "The displacement and strains of a representative case") | 
| _SupVid. 8_ The displacement vector fields and first principal strain values across the carduac cycle for a representative case. |


## Global Results
| <span id="SupFig2">Supplementary Figure 2</span> |
|:--:|
| ![global_all](global_results/individual.png "The normalized volume, displacement magnitude, and strains across the cardiac cycle for each case") |
| _SupFig. 2_ The normalized volume, displacement magnitude, and strains across the cardiac cycle, plotted separately for each case is shown. For each case, the following details are provided: age, sex, condition, minimum volume, maximum volume, pre-activation volume, ejection fraction (EF), and active ejection fraction (aEF) calculated from the 3D images. |


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


| <span id="SupFig3">Supplementary Figure 3</span> |
|:--:|
| ![global_corr](global_results/correlation_matrix.png "Correlation matrix between estimated values from 2-Chamber, 4-Chamber and 3D Views") |
| _SupFig. 3_ The correlation matrix between the left atrial volume (LAV), left ventricular ejection fraction (LVEF), left atrial active ejection fraction (LAaEF), and left atrial ejection fraction (LAEF), as estimated from the 2-chamber (2CH), 4-chamber (4CH), and 3D views. |

| <span id="SupFig4">Supplementary Figure 4</span> |
|:--:|
| ![global_strains](global_results/strains.png "Normalized strains of 2-Chamber, 4-Chamber and 3D Views") |
| _SupFig. 4_ Comparison between the normalized global longitudinal strains from the 2-chamber and 4-chamber, and the mean principal strain values from the 3D view. |


## Regional Results (Atlas-based)
| <span id="SupVid9">Supplementary Video 9</span> |
|:--:|
| ![atlas](regional_results/atlas.gif "Overview of the atlas") |
| _SupVid. 9_ Overview of the constructed atlas from the available healthy population. The columns depict the whole heart view, the left atrium displacement vector field, and the first principal strain. The anterior and posterior views are shown in the first and second rows, respectively. |

| <span id="SupVid10">Supplementary Video 10</span> |
|:--:|
| ![atlas_h1](regional_results/healthy1.gif "Analysis of a healthy case 1 using the atlas") | 
| _SupVid. 10_ The Mahalanobis distance of a healthy case (26M) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid11">Supplementary Video 11</span> |
|:--:|
| ![atlas_h2](regional_results/healthy2.gif "Analysis of a healthy case 2 using the atlas") |
| _SupVid. 11_ The Mahalanobis distance of a healthy case (22F) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid12">Supplementary Video 12</span> |
|:--:|
| ![atlas_h3](regional_results/healthy3.gif "Analysis of a healthy case 3 using the atlas") |
| _SupVid. 12_ The Mahalanobis distance of a healthy case (24F) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid13">Supplementary Video 13</span> |
|:--:|
| ![atlas_p1](regional_results/patient1.gif "Analysis of a patient case 1 using the atlas") |
| _SupVid. 13_ The Mahalanobis distance of a patient case (52F with myocarditis) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid14">Supplementary Video 14</span> |
|:--:|
| ![atlas_p2](regional_results/patient2.gif "Analysis of a patient case 2 using the atlas") |
| _SupVid. 14_ The Mahalanobis distance of a patient case (60M with myocardial infarction) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid15">Supplementary Video 15</span> |
|:--:|
| ![atlas_p3](regional_results/patient3.gif "Analysis of a patient case 3 using the atlas") |
| _SupVid. 15_ The Mahalanobis distance of a patient case (57F with non-ischaemic cardiomyopathy) from the atlas across the cardiac cycle. The first column shows the distance of the displacement vector field, and the second column shows the distance of the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid16">Supplementary Video 16</span> |
|:--:|
| ![atlas_h1_dvf](regional_results/healthy_dvf1.gif "Displacement and strain of a healthy case 1 mapped to the atlas") | 
| _SupVid. 16_ The displacement vector fields and first principal strains of a healthy case (26M) mapped to the atlas across the cardiac cycle. The first column shows the displacement vector field, and the second column shows the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid17">Supplementary Video 17</span> |
|:--:|
| ![atlas_h2_dvf](regional_results/healthy_dvf2.gif "Displacement and strain of a healthy case 2 mapped to the atlas") |
| _SupVid. 17_ The displacement vector fields and first principal strains of a healthy case (22F) mapped to the atlas across the cardiac cycle. The first column shows the displacement vector field, and the second column shows the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid18">Supplementary Video 18</span> |
|:--:|
| ![atlas_h3_dvf](regional_results/healthy_dvf3.gif "Displacement and strain of a healthy case 2 mapped to the atlas") |
| _SupVid. 18_ The displacement vector fields and first principal strains of a healthy case (24F) mapped to the atlas across the cardiac cycle. The first column shows the displacement vector field, and the second column shows the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid19">Supplementary Video 19</span> |
|:--:|
| ![atlas_p1_dvf](regional_results/patient_dvf1.gif "Displacement and strain of a patient case 1 mapped to the atlas") |
| _SupVid. 19_ The displacement vector fields and first principal strains of a patient case (52F with myocarditis) mapped to the atlas across the cardiac cycle. The first column shows the displacement vector field, and the second column shows the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid20">Supplementary Video 20</span> |
|:--:|
| ![atlas_p2_dvf](regional_results/patient_dvf2.gif "Displacement and strain of a patient case 2 mapped to the atlas") |
| _SupVid. 20_ The displacement vector fields and first principal strains of a patient case (60M with myocardial infarction) mapped to the atlas across the cardiac cycle. The first column shows the displacement vector field, and the second column shows the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

| <span id="SupVid21">Supplementary Video 21</span> |
|:--:|
| ![atlas_p3_dvf](regional_results/patient_dvf3.gif "Displacement and strain of a patient case 3 mapped to the atlas") |
| _SupVid. 21_ The displacement vector fields and first principal strains of a patient case (57F with non-ischaemic cardiomyopathy) mapped to the atlas across the cardiac cycle. The first column shows the displacement vector field, and the second column shows the first principal strain. The anterior and posterior views are shown in rows 1 and 2, respectively. |

