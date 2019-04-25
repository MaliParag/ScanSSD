for i in {35..50..5}
do
        python3 /home/psm2208/code/gtdb/stitch_patches.py $i > stitch_$i.out && python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$i/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$i.result &
        python3 /home/psm2208/code/gtdb/stitch_patches.py $((i+1)) > stitch_$((i+1)).out && python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$((i+1)).0/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$((i+1)).result &
        python3 /home/psm2208/code/gtdb/stitch_patches.py $((i+2)) > stitch_$((i+2)).out && python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$((i+2)).0/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$((i+2)).result &
        python3 /home/psm2208/code/gtdb/stitch_patches.py $((i+3)) > stitch_$((i+3)).out && python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$((i+3)).0/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$((i+3)).result &
        python3 /home/psm2208/code/gtdb/stitch_patches.py $((i+4)) > stitch_$((i+4)).out && python3 /home/psm2208/Workspace/IOU_lib/IOUevaluater.py --detections /home/psm2208/code/eval/Train3_Focal_10_25/voting_equal_$((i+4)).0/ --ground_truth /home/psm2208/Workspace/Task3_Detection/Train/GT_math/GT_math/ > voting_equal_$((i+4)).result &

        wait

done