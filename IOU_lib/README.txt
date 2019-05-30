------------------------------------------------------------------------------------------------------------------------
This tool can be used to evaluate IoU metric for predicted math regions against ground truth math regions for each pdf file. For each pdf in test set, generate a csv file which contains predicted bounding box regions. Each line in the csv file corresponds to a bounding box for one math region. It consists of 5 attributes:

page number, x, y, x2, y2 


<detections> contains <pdf_name.csv> files which contains predicted math regions.
<ground_truths> contains <pdf_name.csv> files which contains ground-truth math regions.

For each math region in <ground_truths>, the IoU metric is computed for every math region in <detecions> (per page) and returns a sorted list of (ground_truth: IOU score,detection) in descending order. This information is saved for each page in a folder named "IOU_scores_pages".

Usage:
python3 IOUevaluater.py --detections <detectionsDir> --ground_truth <ground_truthsDir>

------------------------------------------------------------------------------------------------------------------------
Acknowledgements
Thanks to Rafael Padilla.
https://github.com/rafaelpadilla/Object-Detection-Metrics
