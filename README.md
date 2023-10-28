# An Emotional Journey: Detecting Emotion Trajectories in Dutch Customer Service Dialogues

This repository contains the code for the experiment of the following paper:

```
@inproceedings{labat-etal-2022-emotional,
    title = "An Emotional Journey: Detecting Emotion Trajectories in {D}utch Customer Service Dialogues",
    author = "Labat, Sofie  and
      Hadifar, Amir  and
      Demeester, Thomas  and
      Hoste, Veronique",
    booktitle = "Proceedings of the Eighth Workshop on Noisy User-generated Text (W-NUT 2022)",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wnut-1.12",
    pages = "106--112",
    abstract = "The ability to track fine-grained emotions in customer service dialogues has many real-world applications, but has not been studied extensively. This paper measures the potential of prediction models on that task, based on a real-world dataset of Dutch Twitter conversations in the domain of customer service. We find that modeling emotion trajectories has a small, but measurable benefit compared to predictions based on isolated turns. The models used in our study are shown to generalize well to different companies and economic sectors.",
}
```

## Install dependencies

`pip install -r requirement.txt`

## Running experiment


Majority class basline: 
`sh script/run_mb_basline.sh`

SVM baseline:
`sh script/run_svm_baseline.sh`

Sector experiment:
`sh script/run_task.sh`

CRF experiment:
`sh script/run_crf_exp.sh`



## Visualization
See `visualizations` folder

![](https://github.com/hadifar/DutchEmotionDetection/blob/main/visualizations/wcrf_wocrf.png)
