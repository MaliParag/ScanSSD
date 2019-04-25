for i in {30..55..1}
do
        python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$i.0/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$i.result &
        #python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$((i+1)).0/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$((i+1)).result &
        #python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$((i+2)).0/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$((i+2)).result &
        #python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$((i+3)).0/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$((i+3)).result &
        #python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$((i+4)).0/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$((i+4)).result &

        wait

done