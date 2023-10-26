from hashlib import new
from itertools import count
import numpy as np
import torch
import time
from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
from queue import Queue

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_dist = 0.2
        max_cosine_distance = max_dist
        nn_budget = 100
        max_age = 50
        self.tracker = Tracker(max_dist, nn_budget, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img, queue, frame, cam_index=0):
        self.height, self.width = ori_img.shape[:2]

        # generate detections (detections is a list with class Detection which contain 1.tlwh 2.confidence 3.feature)
        #                tlwh[1]         tlwh[2]       
        # detections = [ confidence[1], confidence[2], ...... ]
        #                feature[1]      feature[2]
  
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict(cam_index)
        self.tracker.update(detections, cam_index)

        '''
        # original code
        for track in self.tracker.tracks:
            # output bbox identities
            outputs = []
            for k, _ in enumerate(track):
                if not track[k].is_confirmed() or track[k].time_since_update > 1:
                    continue
                box = track[k].to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track[k].track_id
                outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
            if len(outputs) > 0:
                outputs = np.stack(outputs,axis=0)

            queue.put(outputs)
        '''
        '''
        # Version of only solve occlusion problem
        outputs = [[] for _ in range(2)]
        for track in self.tracker.tracks[cam_index]:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs[cam_index].append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs[cam_index]) > 0:
            outputs[cam_index] = np.stack(outputs[cam_index],axis=0)
        '''

        ## ADD CODE##
        # Version of solve parking in
        ##############################################################
        """
        # # background tracking results
        # if cam_index == 0:
        #     queue.put(self.tracker.tracks[0])

        # '''
        # TODO:
        #     1. Determine the obj in original frame is fg/bg obj
        #     2. Deal with the case that a car is parking in
        #     3. Deal with the case that a car is leaving the parking lot
        # '''
        # if cam_index == 1:
        #     # the real background/foreground objects that going to plot on the video
        #     bg_obj = []
        #     fg_obj = []

        #     # the background objects in orginal frame
        #     InOri_bg_obj = []

        #     # The car that already parked well
        #     parked = []

        #     # The car that is leaving
        #     leaving = []

        #     # Get the trackers in bg/ori frame
        #     self.tracker.tracks[0] = queue.get()


        #     # Traverse through all bboxes of background frame
        #     for bg_indice, bg_track in enumerate(self.tracker.tracks[0]):

        #         # For 3.
        #         # To find biggest bg_overlap if < 0.35, means the car is leaving
        #         max_overlap = 0.0
        #         leave = []

        #         # Traverse through all bboxes of original frame
        #         for ori_indice, ori_track in enumerate(self.tracker.tracks[1]):
        #             # Transform the bbox into(x1, y1,x2, y2)
        #             box = ori_track.to_tlwh()
        #             ori_bb = self._tlwh_to_xyxy(box)
        #             box = bg_track.to_tlwh()
        #             bg_bb = self._tlwh_to_xyxy(box)
                    
        #             # Calculate the top-left and bottom-right coordinaries of two bbox
        #             ori_bb1 = [ori_bb[0], ori_bb[1]] # origin obj bbox top-left (x1, x2)
        #             ori_bb2 = [ori_bb[2], ori_bb[3]] # origin obj bbox bottom-right (x2, y2)
        #             bg_bb1 = [bg_bb[0], bg_bb[1]] # background obj bbox top-left (x1, x2)
        #             bg_bb2 = [bg_bb[2], bg_bb[3]] # background obj bbox bottom-right (x2, y2)
                    
        #             # determine the coordinates of the intersection rectangle
        #             x_left = max(ori_bb1[0], bg_bb1[0])
        #             y_top = max(ori_bb1[1], bg_bb1[1])
        #             x_right = min(ori_bb2[0], bg_bb2[0])
        #             y_bottom = min(ori_bb2[1], bg_bb2[1])
                    
        #             # In case that there is no intersection
        #             if x_right < x_left or y_bottom < y_top:
        #                 overlap_ori = 0.0
        #                 overlap_bg = 0.0
                    
        #             else:
        #                 # Calculate intersection
        #                 intersection_area = (x_right - x_left) * (y_bottom - y_top)

        #                 # Calculate overlap of (1)bg_bb_area and intersection_area (2)fg_bbox_area and intersection_area
        #                 bg_bb_area = (bg_bb2[0] - bg_bb1[0]) * (bg_bb2[1] - bg_bb1[1])
        #                 ori_bb_area = (ori_bb2[0] - ori_bb1[0]) * (ori_bb2[1] - ori_bb1[1])
        #                 if bg_bb_area == 0:
        #                     overlap_bg = 0
        #                 else:
        #                     overlap_bg = intersection_area / float(bg_bb_area)
        #                 if ori_bb_area == 0:
        #                     overlap_ori = 0
        #                 else:
        #                     overlap_ori = intersection_area / float(ori_bb_area)

        #                 # For 1.
        #                 if overlap_ori > 0.8:
        #                     InOri_bg_obj.append(ori_indice)
        #                 # for 2.
        #                 if overlap_ori > 0.8 and overlap_bg > 0.8:
        #                     parked.append([ori_indice, bg_indice])
        #                 # for 3.
        #                 if (self.tracker.tracks[0][bg_indice].is_leave == False) and (self.tracker.tracks[1][ori_indice].fg_is_leave == False):
        #                     if overlap_bg > max_overlap:
        #                         max_overlap = overlap_bg
        #                         leave = [ori_indice, bg_indice]

        #         # For 3.
        #         # If the max_overlap is < 0.35 means the car is leaving
        #         if max_overlap != 0 and max_overlap < 0.35:
        #             leaving.append(leave)

        #     ## 1. Determine the objs in original frame is a fg or bg objs
        #     for indice in range(len(self.tracker.tracks[1])):
        #         if indice not in InOri_bg_obj:
        #             fg_obj.append(self.tracker.tracks[1][indice])
            
        #     ## 2. Deal with the case that a car is parking in
        #     if len(parked) != 0:
        #         for pair in parked:
        #             ori_index = pair[0]
        #             bg_index = pair[1]

        #             if self.tracker.tracks[0][bg_index].is_parked is False:
        #                 # Only change at the first time
        #                 self.tracker.tracks[0][bg_index].is_parked = True
        #                 # Give the foreground id to the just appear background bbox
        #                 new_id = self.tracker.tracks[1][ori_index].track_id
        #                 old_id = self.tracker.tracks[0][bg_index].track_id
        #                 # Give the foreground id to the just appear background bbox
        #                 self.tracker.tracks[0][bg_index].track_id = self.tracker.tracks[1][ori_index].track_id
        #                 # Change the samples in tracker.metric too!!!!!
        #                 if self.tracker.metric[0].samples.get(old_id) != None:
        #                     self.tracker.metric[0].samples[new_id] = self.tracker.metric[0].samples.pop(old_id)

        #     ## 3. Deal with the case that a car is leaving the parking lot, none of the ori_bbox is overlap with the bg_bbox exceed 0.35
        #     if len(leaving) != 0:
        #         for pair in leaving:
        #             ori_index = pair[0]
        #             bg_index = pair[1]
        #             '''
        #             if bg_tracks[bg_index].is_leave is False:
        #                 ori_tracks[ori_indiex].track_id = bg_tracks[bg_index].track_id
        #                 bg_tracks[bg_index].is_leave = True
        #             '''
        #             '''
        #             temp = self.tracker.tracks[1][ori_index].track_id
        #             self.tracker.tracks[1][ori_index].track_id = self.tracker.tracks[0][bg_index].track_id
        #             self.tracker.tracks[0][bg_index].track_id = temp
        #             '''
        #             '''
        #             self.tracker.tracks[1][ori_index].fg_new_id = self.tracker.tracks[0][bg_index].track_id
        #             self.tracker.tracks[1][ori_index].fg_is_leave = True
        #             self.tracker.tracks[0][bg_index].is_leave = True
        #             '''
        #             # Only change at the first time
        #             self.tracker.tracks[0][bg_index].is_leave = True
        #             self.tracker.tracks[1][ori_index].fg_is_leave = True
        #             # Give the background id to the just appear foreground bbox that is going to leave
        #             new_id = self.tracker.tracks[0][bg_index].track_id
        #             old_id = self.tracker.tracks[1][ori_index].track_id
        #             # Give the background id to the just appear foreground bbox that is going to leave
        #             self.tracker.tracks[1][ori_index].track_id = self.tracker.tracks[0][bg_index].track_id
        #             # Change the samples in tracker.metric too!!!!!
        #             if self.tracker.metric[1].samples.get(old_id) != None:
        #                 self.tracker.metric[1].samples[new_id] = self.tracker.metric[1].samples.pop(old_id)


        #     # Determine the bg objs is still parking or not
        #     for track in self.tracker.tracks[0]:
        #         if track.is_leave == False:
        #             bg_obj.append(track)

        #     # Output the final results
        #     bg_outputs = []
        #     fg_outputs = []
        #     for track in bg_obj:
        #         if not track.is_confirmed() or track.time_since_update > 1:
        #             continue
        #         box = track.to_tlwh()
        #         x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
        #         track_id = track.track_id
        #         bg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        #     if len(bg_outputs) > 0:
        #         bg_outputs = np.stack(bg_outputs,axis=0)

        #     for track in fg_obj:
        #         if not track.is_confirmed() or track.time_since_update > 1:
        #             continue
        #         box = track.to_tlwh()
        #         x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
        #         '''
        #         if track.fg_is_leave is True:
        #             track_id = track.fg_new_id
        #         else:
        #             track_id = track.track_id
        #         '''   
        #         track_id = track.track_id
        #         fg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        #     if len(fg_outputs) > 0:
        #         fg_outputs = np.stack(fg_outputs,axis=0)

        #     # Put the result into queue to return
        #     queue.put(bg_outputs)
        #     queue.put(fg_outputs)
        """
        ##############################################################
        
        # Ver.2
        ##############################################################
        """
        # background tracking results
        if cam_index == 0:
            queue.put(self.tracker.tracks[0])
        
        '''
        <DEFINITIONS>
        foreground objects: The bboxes of cars that are moving in the current frame
        background objects: The bboxes of cars that are parked well in the current frame 
                            (1) The bboxes of parked cars in bg_tracks
                            (2) The bboxes that is moving because of occlusion in ori_tracks
        TODO:
            1. Determine the obj in original frame is fg/bg obj
            2. Deal with the case that a car is parking in
            3. Deal with the case that a car is leaving the parking lot
        '''
        
        
        if cam_index == 1:
            # The real background/foreground objects that going to plot on the video
            bg_objs = []
            fg_objs = []

            # the background objects in orginal frame (we don't want to plot)
            InOri_bg_objs = []

            # The car that already parked well
            parked = []

            # The car that its foreground is totally leave the parking lot
            bg_is_leave = []

            # Get the tracks informations in background frame
            self.tracker.tracks[0] = queue.get()

            # Traverse through all bboxes of background frame
            for bg_indice, bg_track in enumerate(self.tracker.tracks[0]):

                # For 3.
                # To find biggest bg_overlap, if < 0.35, means there is no ori_bbox overlap with this bg_bbox, so this car is already left
                max_overlap = 0.0
                leave = None

                # Traverse through all bboxes of original frame
                for ori_indice, ori_track in enumerate(self.tracker.tracks[1]):
                    # Transform the bbox into: (x1, y1, x2, y2)
                    box = ori_track.to_tlwh()
                    ori_bb = self._tlwh_to_xyxy(box)
                    box = bg_track.to_tlwh()
                    bg_bb = self._tlwh_to_xyxy(box)
                    
                    # Calculate the top-left and bottom-right coordinaries of two bbox
                    ori_bb1 = [ori_bb[0], ori_bb[1]] # origin obj bbox top-left (x1, x2)
                    ori_bb2 = [ori_bb[2], ori_bb[3]] # origin obj bbox bottom-right (x2, y2)
                    bg_bb1 = [bg_bb[0], bg_bb[1]]    # background obj bbox top-left (x1, x2)
                    bg_bb2 = [bg_bb[2], bg_bb[3]]    # background obj bbox bottom-right (x2, y2)
                    
                    # determine the coordinates of the intersection rectangle
                    x_left = max(ori_bb1[0], bg_bb1[0])
                    y_top = max(ori_bb1[1], bg_bb1[1])
                    x_right = min(ori_bb2[0], bg_bb2[0])
                    y_bottom = min(ori_bb2[1], bg_bb2[1])
                    
                    # In case that there is no intersection
                    if x_right < x_left or y_bottom < y_top:
                        overlap_ori = 0.0
                        overlap_bg = 0.0
                    
                    else:
                        # Calculate intersection
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)

                        # Calculate overlap of (1)bg_bb_area and intersection_area (2)fg_bbox_area and intersection_area
                        bg_bb_area = (bg_bb2[0] - bg_bb1[0]) * (bg_bb2[1] - bg_bb1[1])
                        ori_bb_area = (ori_bb2[0] - ori_bb1[0]) * (ori_bb2[1] - ori_bb1[1])
                        if bg_bb_area == 0:
                            overlap_bg = 0
                        else:
                            overlap_bg = intersection_area / float(bg_bb_area)
                        if ori_bb_area == 0:
                            overlap_ori = 0
                        else:
                            overlap_ori = intersection_area / float(ori_bb_area)

                        # For 1.
                        if overlap_ori > 0.8:
                            InOri_bg_objs.append(ori_indice)
                        # for 2.
                        if overlap_ori > 0.8 and overlap_bg > 0.8:
                            parked.append([ori_indice, bg_indice])
                        # for 3.
                        if overlap_bg > max_overlap:
                            max_overlap = overlap_bg
                            leave = bg_indice

                # For 3.
                # If the max_overlap is < 0.35 means the car of this bg is left
                if max_overlap != 0 and max_overlap < 0.35:
                    bg_is_leave.append(leave)

            ## 1. Determine the objs in original frame is a fg or bg objs
            for indice in range(len(self.tracker.tracks[1])):
                if indice not in InOri_bg_objs:
                    fg_objs.append(self.tracker.tracks[1][indice])

            # Put all bg_track into bg_obj (we will decide witch to plot later)
            for bg_track in self.tracker.tracks[0]:
                bg_objs.append(bg_track)

            ## 2. Deal with the case that a car is parking in
            if len(parked) != 0:
                for pair in parked:
                    ori_index = pair[0]
                    bg_index = pair[1]
                    
                    # When the bg_box just appear
                    if self.tracker.tracks[0][bg_index].is_parked is False:
                        # Only change at the first time
                        self.tracker.tracks[0][bg_index].is_parked = True
                        # Give the foreground id to the just appear background bbox
                        new_id = self.tracker.tracks[1][ori_index].track_id
                        old_id = self.tracker.tracks[0][bg_index].track_id
                        # Give the foreground id to the just appear background bbox
                        self.tracker.tracks[0][bg_index].track_id = self.tracker.tracks[1][ori_index].track_id
                        # Change the samples in tracker.metric too!!!!!
                        if self.tracker.metric[0].samples.get(old_id) != None:
                            self.tracker.metric[0].samples[new_id] = self.tracker.metric[0].samples.pop(old_id)

                    # Always keep the parked car's fg_id is the same with its bg_id 
                    if self.tracker.tracks[0][bg_index].is_parked is True:
                        new_id = self.tracker.tracks[0][bg_index].track_id
                        old_id = self.tracker.tracks[1][ori_index].track_id
                        # Give the background id to its foreground
                        self.tracker.tracks[1][ori_index].track_id = self.tracker.tracks[0][bg_index].track_id
                        # Change the samples in tracker.metric too!!!!!
                        if self.tracker.metric[1].samples.get(old_id) != None:
                            self.tracker.metric[1].samples[new_id] = self.tracker.metric[1].samples.pop(old_id)

            # If a car is in fg_objs (going to plot), and its background is exist, then plot the fg_obj, there is two cases:
            # (1) The parked car is leaving, but not totally leave the scene yet
            # (2) The original bbox jittering cause by occlusion or people opening the car's door and so on
            for fg_track in fg_objs:
                for bg_track in bg_objs:
                    if fg_track.track_id == bg_track.track_id:
                        bg_objs.remove(bg_track)


            ## 3. Deal with the case that a car is totally left the parking lot, none of the ori_bbox is overlap with the bg_bbox exceed 0.35
            if len(bg_is_leave) != 0:
                for bg_index in bg_is_leave:
                    if self.tracker.tracks[0][bg_indice].is_leave == False:
                        # Only change at the first time
                        self.tracker.tracks[0][bg_index].is_leave = True


            # Determine the bg objs is still parking or not
            for bg_track in bg_objs:
                if bg_track.is_leave == True:
                    bg_objs.remove(bg_track)

            # Output the final results
            bg_outputs = []
            fg_outputs = []
            for track in bg_objs:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                bg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
            if len(bg_outputs) > 0:
                bg_outputs = np.stack(bg_outputs,axis=0)

            for track in fg_objs:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                fg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
            if len(fg_outputs) > 0:
                fg_outputs = np.stack(fg_outputs,axis=0)

            # Put the result into queue to return
            queue.put(bg_outputs)
            queue.put(fg_outputs)
        """
        ##############################################################

        # Ver.3
        ##############################################################
        """
        # background tracking results
        if cam_index == 0:
            queue.put(self.tracker.tracks[0])
        '''
        <DEFINITIONS>
        foreground objects: The bboxes of cars that are moving in the current frame
        background objects: The bboxes of cars that are parked well in the current frame 
                            (1) The bboxes of parked cars in bg_tracks
                            (2) The bboxes that is moving because of occlusion in ori_tracks
        TODO:
            1. Determine the obj in original frame is fg/bg obj
            2. Deal with the case that a car is parking in
            3. Deal with the case that a car is leaving the parking lot
        '''
        
        if cam_index == 1:
            # The real background/foreground objects that are going to be plot in the video
            bg_objs = []
            fg_objs = []

            # the background objects in orginal frame (those we don't want to plot)
            InOri_bg_objs = []
            InOri_temp_miss = []
            # InOri_bg_obj_ID = []

            # The car that already parked well
            parked = []

            # The car that its foreground has completely left the parking lot
            bg_is_leave = []

            # Get the tracks informations in background frame
            self.tracker.tracks[0] = queue.get()

            # fg_id_initial = []
            # for fg_track in self.tracker.tracks[1]:
            #     fg_id_initial.append(fg_track.track_id)
            # print(f'fg_id_initial: {fg_id_initial}')

            # Traverse through all bboxes of background frame
            for bg_indice, bg_track in enumerate(self.tracker.tracks[0]):

                # For 3.
                # To find biggest bg_overlap, if < 0.35, means there is no ori_bbox overlap with this bg_bbox, so this car is already left
                max_overlap = 0.0
                leave = None

                # Traverse through all bboxes of original frame
                for ori_indice, ori_track in enumerate(self.tracker.tracks[1]):
                    # Transform the bbox into: (x1, y1, x2, y2)
                    box = ori_track.to_tlwh()
                    ori_bb = self._tlwh_to_xyxy(box)
                    box = bg_track.to_tlwh()
                    bg_bb = self._tlwh_to_xyxy(box)
                    
                    # Calculate the top-left and bottom-right coordinaries of two bbox
                    ori_bb1 = [ori_bb[0], ori_bb[1]] # origin obj bbox top-left (x1, y2)
                    ori_bb2 = [ori_bb[2], ori_bb[3]] # origin obj bbox bottom-right (x2, y2)
                    bg_bb1 = [bg_bb[0], bg_bb[1]]    # background obj bbox top-left (x1, x2)
                    bg_bb2 = [bg_bb[2], bg_bb[3]]    # background obj bbox bottom-right (x2, y2)
                    
                    # determine the coordinates of the intersection rectangle
                    x_left = max(ori_bb1[0], bg_bb1[0])
                    y_top = max(ori_bb1[1], bg_bb1[1])
                    x_right = min(ori_bb2[0], bg_bb2[0])
                    y_bottom = min(ori_bb2[1], bg_bb2[1])
                    
                    # bg_bb_area = (bg_bb2[0] - bg_bb1[0]) * (bg_bb2[1] - bg_bb1[1])
                    # ori_bb_area = (ori_bb2[0] - ori_bb1[0]) * (ori_bb2[1] - ori_bb1[1])

                    # if ori_track.track_id == 1:
                    #     print(f'1_ori_bb_area: {ori_bb_area}')

                    # In case that there is no intersection
                    if x_right < x_left or y_bottom < y_top:
                        overlap_ori = 0.0
                        overlap_bg = 0.0
                        if ori_track.track_id == bg_track.track_id:
                            # print(f'InOri_temp_miss: {ori_track.track_id}')
                            InOri_temp_miss.append(ori_indice)
                    
                    else:
                        # Calculate intersection
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)

                        # Calculate overlap of (1)bg_bb_area and intersection_area (2)fg_bbox_area and intersection_area
                        bg_bb_area = (bg_bb2[0] - bg_bb1[0]) * (bg_bb2[1] - bg_bb1[1])
                        ori_bb_area = (ori_bb2[0] - ori_bb1[0]) * (ori_bb2[1] - ori_bb1[1])
                        if bg_bb_area == 0:
                            overlap_bg = 0
                        else:
                            overlap_bg = intersection_area / float(bg_bb_area)
                        if ori_bb_area == 0:
                            overlap_ori = 0
                        else:
                            overlap_ori = intersection_area / float(ori_bb_area)

                        # if ori_track.track_id == 1:
                        #     print(f'1_overlap_ori: {overlap_ori}')
                            # print(f'1_overlap_bg: {overlap_bg}')

                    # For 1.
                    if overlap_ori > 0.8:
                        InOri_bg_objs.append(ori_indice)
                        # InOri_bg_obj_ID.append(ori_track.track_id)

                    # for 2.
                    if overlap_ori > 0.8 and overlap_bg > 0.8:
                        parked.append([ori_indice, bg_indice])

                    # for 3.
                    if overlap_bg > max_overlap:
                        max_overlap = overlap_bg
                        leave = bg_indice


                    # if ori_track.track_id == 1:
                    #     print(f'1_overlap_ori_BEF: {overlap_ori}')


                # For 3.
                # If the max_overlap is < 0.35 means the car of this bg is left
                if max_overlap != 0 and max_overlap < 0.35:
                    bg_is_leave.append(leave)

            ## 2. Deal with the case that a car is parking in
            if len(parked) != 0:
                for pair in parked:
                    ori_index = pair[0]
                    bg_index = pair[1]
                    
                    # When the bg_box just appear
                    if self.tracker.tracks[0][bg_index].is_parked is False:
                        # Only change at the first time
                        self.tracker.tracks[0][bg_index].is_parked = True
                        # Give the foreground id to the just appear background bbox
                        new_id = self.tracker.tracks[1][ori_index].track_id
                        old_id = self.tracker.tracks[0][bg_index].track_id
                        # Give the foreground id to the just appear background bbox
                        self.tracker.tracks[0][bg_index].track_id = self.tracker.tracks[1][ori_index].track_id
                        # Change the samples in tracker.metric too!!!!!
                        if self.tracker.metric[0].samples.get(old_id) != None:
                            self.tracker.metric[0].samples[new_id] = self.tracker.metric[0].samples.pop(old_id)

                    # Always keep the parked car's fg_id is the same with its bg_id 
                    if self.tracker.tracks[0][bg_index].is_parked is True:
                        new_id = self.tracker.tracks[0][bg_index].track_id
                        old_id = self.tracker.tracks[1][ori_index].track_id

                        # if new_id == 1:
                            # print(f'ori_id: {old_id}')
                            # print(f'bg_id: {new_id}')

                        # Give the background id to its foreground
                        self.tracker.tracks[1][ori_index].track_id = self.tracker.tracks[0][bg_index].track_id
                        # Change the samples in tracker.metric too!!!!!
                        if self.tracker.metric[1].samples.get(old_id) != None:
                            self.tracker.metric[1].samples[new_id] = self.tracker.metric[1].samples.pop(old_id)


            ## 3. Deal with the case that a car is totally left the parking lot, none of the ori_bbox is overlap with the bg_bbox exceed 0.35
            if len(bg_is_leave) != 0:
                for bg_index in bg_is_leave:
                    if self.tracker.tracks[0][bg_index].is_leave is False:
                        # Only change at the first time
                        self.tracker.tracks[0][bg_index].is_leave = True

            
            # fg_id_bef_rmbg = []
            # for fg_track in self.tracker.tracks[1]:
            #     fg_id_bef_rmbg.append(fg_track.track_id)
            # print(f'fg_id_bef_rmbg: {fg_id_bef_rmbg}')

            
            ## 1. Determine the objs in original frame is a fg or bg objs
            for ori_indice, ori_track in enumerate(self.tracker.tracks[1]):
                if ori_indice not in InOri_bg_objs and ori_indice not in InOri_temp_miss:
                    fg_objs.append(ori_track)


            # print(f'InOri_bg_objs: {InOri_bg_objs}')
            # print(f'InOri_bg_obj_ID: {InOri_bg_obj_ID}')
            
            # fg_id_final = []
            # for fg_track in fg_objs:
            #     fg_id_final.append(fg_track.track_id)
            # print(f'fg_id_final: {fg_id_final}')


            
            # Put all bg_track into bg_obj (we will decide witch to plot later)
            for bg_track in self.tracker.tracks[0]:
                bg_objs.append(bg_track)


            # If a car is in fg_objs (going to plot), and its background is exist, then plot the fg_obj, there is two cases:
            # (1) The parked car is leaving, but not totally leave the scene yet
            # (2) The original bbox jittering cause by occlusion or people opening the car's door and so on
            for fg_track in fg_objs:
                for bg_track in bg_objs:
                    if fg_track.track_id == bg_track.track_id:
                        # print(f'fg_leave_bg_id: {bg_track.track_id}')
                        bg_objs.remove(bg_track)

            # Determine the bg objs is still parking or not
            for bg_track in bg_objs:
                if bg_track.is_leave == True:
                    # print(f'is_leave_bg_id: {bg_track.track_id}')
                    bg_objs.remove(bg_track)


            # bg_id = []
            # for bg_track in bg_objs:
            #     bg_id.append(bg_track.track_id)
            # print(f'bg_id: {bg_id}')


            # Output the final results
            bg_outputs = []
            fg_outputs = []
            for track in bg_objs:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                bg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
            if len(bg_outputs) > 0:
                bg_outputs = np.stack(bg_outputs,axis=0)

            for track in fg_objs:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                fg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
            if len(fg_outputs) > 0:
                fg_outputs = np.stack(fg_outputs,axis=0)

            # Put the result into queue to return
            queue.put(bg_outputs)
            queue.put(fg_outputs)
        """
        ##############################################################

        # Ver.4
        ##############################################################
        """
        # background tracking results
        if cam_index == 0:
            queue.put(self.tracker.tracks[0])
        '''
        <DEFINITIONS>
        foreground objects: The bboxes of cars that are moving in the current frame
        background objects: The bboxes of cars that are parked well in the current frame 
                            (1) The bboxes of parked cars in bg_tracks
                            (2) The bboxes that is moving because of occlusion in ori_tracks
        TODO:
            1. Determine the obj in original frame is fg/bg obj
            2. Deal with the case that a car is parking in
            3. Deal with the case that a car is leaving the parking lot
        '''
        
        if cam_index == 1:
            # The real background/foreground objects that are going to be plot in the video
            bg_objs = []
            fg_objs = []

            # the background objects in orginal frame (those we don't want to plot)
            InOri_bg_objs = []
            InOri_temp_miss = []
            # InOri_bg_obj_ID = []

            # The car that already parked well
            parked = []

            # The car that its foreground has completely left the parking lot
            bg_is_leave = []

            # Get the tracks informations in background frame
            self.tracker.tracks[0] = queue.get()

            # bg_tracks = []
            # for bg_track in self.tracker.tracks[0]:
            #     if not bg_track.is_confirmed() or bg_track.time_since_update > 1:
            #         continue
            #     bg_tracks.append(bg_track)

            # ori_tracks = []
            # for ori_track in self.tracker.tracks[1]:
            #     if not ori_track.is_confirmed() or ori_track.time_since_update > 1:
            #         continue
            #     ori_tracks.append(ori_track)

            # ori_id_initial = []
            # for ori_track in ori_tracks:
            #     ori_id_initial.append(ori_track.track_id)
            # print(f'ori_id_initial: {ori_id_initial}')

            # bg_id_initial = []
            # for bg_track in bg_tracks:
            #     bg_id_initial.append(bg_track.track_id)
            # print(f'bg_id_initial: {bg_id_initial}')

            # fg_id_initial = []
            # for fg_track in self.tracker.tracks[1]:
            #     fg_id_initial.append(fg_track.track_id)
            # print(f'fg_id_initial: {fg_id_initial}')

            # Traverse through all bboxes of background frame
            for bg_indice, bg_track in enumerate(self.tracker.tracks[0]):
                if not bg_track.is_confirmed() or bg_track.time_since_update > 1: 
                    continue

                # For 3.
                # To find biggest bg_overlap, if < 0.35, means there is no ori_bbox overlap with this bg_bbox, so this car is already left
                max_overlap = 0.0
                leave = None

                # Traverse through all bboxes of original frame
                for ori_indice, ori_track in enumerate(self.tracker.tracks[1]):
                    if not ori_track.is_confirmed() or ori_track.time_since_update > 1: 
                        continue

                    # Transform the bbox into: (x1, y1, x2, y2)
                    box = ori_track.to_tlwh()
                    ori_bb = self._tlwh_to_xyxy(box)
                    box = bg_track.to_tlwh()
                    bg_bb = self._tlwh_to_xyxy(box)
                    
                    # Calculate the top-left and bottom-right coordinaries of two bbox
                    ori_bb1 = [ori_bb[0], ori_bb[1]] # origin obj bbox top-left (x1, y2)
                    ori_bb2 = [ori_bb[2], ori_bb[3]] # origin obj bbox bottom-right (x2, y2)
                    bg_bb1 = [bg_bb[0], bg_bb[1]]    # background obj bbox top-left (x1, x2)
                    bg_bb2 = [bg_bb[2], bg_bb[3]]    # background obj bbox bottom-right (x2, y2)
                    
                    # determine the coordinates of the intersection rectangle
                    x_left = max(ori_bb1[0], bg_bb1[0])
                    y_top = max(ori_bb1[1], bg_bb1[1])
                    x_right = min(ori_bb2[0], bg_bb2[0])
                    y_bottom = min(ori_bb2[1], bg_bb2[1])
                    
                    # bg_bb_area = (bg_bb2[0] - bg_bb1[0]) * (bg_bb2[1] - bg_bb1[1])
                    # ori_bb_area = (ori_bb2[0] - ori_bb1[0]) * (ori_bb2[1] - ori_bb1[1])

                    # if ori_track.track_id == 1:
                    #     print(f'1_ori_bb_area: {ori_bb_area}')

                    # In case that there is no intersection
                    if x_right < x_left or y_bottom < y_top:
                        overlap_ori = 0.0
                        overlap_bg = 0.0
                        # if ori_track.track_id == bg_track.track_id:
                            # print(f'InOri_temp_miss: {ori_track.track_id}')
                            # InOri_temp_miss.append(ori_indice)
                    
                    else:
                        # Calculate intersection
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)

                        # Calculate overlap of (1)bg_bb_area and intersection_area (2)fg_bbox_area and intersection_area
                        bg_bb_area = (bg_bb2[0] - bg_bb1[0]) * (bg_bb2[1] - bg_bb1[1])
                        ori_bb_area = (ori_bb2[0] - ori_bb1[0]) * (ori_bb2[1] - ori_bb1[1])
                        if bg_bb_area == 0.0:
                            overlap_bg = 0.0
                        else:
                            overlap_bg = intersection_area / float(bg_bb_area)
                        if ori_bb_area == 0.0:
                            overlap_ori = 0.0
                        else:
                            overlap_ori = intersection_area / float(ori_bb_area)

                    # if ori_track.track_id == 4:
                    #     print(f'bg_id: {bg_track.track_id}')
                    #     print(f'4_overlap_ori: {overlap_ori}')
                    #     # print(f'1_overlap_bg: {overlap_bg}')

                    # For 1.
                    if overlap_ori > 0.8:
                        InOri_bg_objs.append(ori_indice)
                        # InOri_bg_obj_ID.append(ori_track.track_id)

                    # for 2.
                    if overlap_ori > 0.8 and overlap_bg > 0.8:
                        parked.append([ori_indice, bg_indice])

                    # for 3.
                    if overlap_bg > max_overlap:
                        max_overlap = overlap_bg
                        leave = bg_indice


                    # if ori_track.track_id == 1:
                    #     print(f'1_overlap_ori_BEF: {overlap_ori}')


                # For 3.
                # If the max_overlap is < 0.35 means the car of this bg is left
                if max_overlap != 0 and max_overlap < 0.35:
                    bg_is_leave.append(leave)

            ## 2. Deal with the case that a car is parking in
            if len(parked) != 0:
                for pair in parked:
                    ori_index = pair[0]
                    bg_index = pair[1]
                    
                    # When the bg_box just appear
                    if self.tracker.tracks[0][bg_index].is_parked is False:
                        # Only change at the first time
                        self.tracker.tracks[0][bg_index].is_parked = True
                        # Give the foreground id to the just appear background bbox
                        new_id = self.tracker.tracks[1][ori_index].track_id
                        old_id = self.tracker.tracks[0][bg_index].track_id
                        # Give the foreground id to the just appear background bbox
                        self.tracker.tracks[0][bg_index].track_id = self.tracker.tracks[1][ori_index].track_id
                        # Change the samples in tracker.metric too!!!!!
                        if self.tracker.metric[0].samples.get(old_id) != None:
                            self.tracker.metric[0].samples[new_id] = self.tracker.metric[0].samples.pop(old_id)

                    # Always keep the parked car's fg_id is the same with its bg_id 
                    if self.tracker.tracks[0][bg_index].is_parked is True:
                        new_id = self.tracker.tracks[0][bg_index].track_id
                        old_id = self.tracker.tracks[1][ori_index].track_id

                        # if new_id == 1:
                            # print(f'ori_id: {old_id}')
                            # print(f'bg_id: {new_id}')

                        # Give the background id to its foreground
                        self.tracker.tracks[1][ori_index].track_id = self.tracker.tracks[0][bg_index].track_id
                        # Change the samples in tracker.metric too!!!!!
                        if self.tracker.metric[1].samples.get(old_id) != None:
                            self.tracker.metric[1].samples[new_id] = self.tracker.metric[1].samples.pop(old_id)


            ## 3. Deal with the case that a car is totally left the parking lot, none of the ori_bbox is overlap with the bg_bbox exceed 0.35
            if len(bg_is_leave) != 0:
                for bg_index in bg_is_leave:
                    if self.tracker.tracks[0][bg_index].is_leave is False:
                        # Only change at the first time
                        self.tracker.tracks[0][bg_index].is_leave = True

            
            # fg_id_bef_rmbg = []
            # for fg_track in self.tracker.tracks[1]:
            #     fg_id_bef_rmbg.append(fg_track.track_id)
            # print(f'fg_id_bef_rmbg: {fg_id_bef_rmbg}')

            
            ## 1. Determine the objs in original frame is a fg or bg objs
            for ori_indice, ori_track in enumerate(self.tracker.tracks[1]):
                if not ori_track.is_confirmed() or ori_track.time_since_update > 1:
                    continue
                if ori_indice not in InOri_bg_objs: #and ori_indice not in InOri_temp_miss
                    fg_objs.append(ori_track)


            # print(f'InOri_bg_objs: {InOri_bg_objs}')
            # print(f'InOri_bg_obj_ID: {InOri_bg_obj_ID}')
            
            # fg_id_final = []
            # for fg_track in fg_objs:
            #     fg_id_final.append(fg_track.track_id)
            # print(f'fg_id_final: {fg_id_final}')


            # Put all bg_track into bg_obj (we will decide witch to plot later)
            for bg_track in self.tracker.tracks[0]:
                if not bg_track.is_confirmed() or bg_track.time_since_update > 1: 
                    continue
                bg_objs.append(bg_track)


            # If a car is in fg_objs (going to plot), and its background is exist, then plot the fg_obj, there is two cases:
            # (1) The parked car is leaving, but not totally leave the scene yet
            # (2) The original bbox jittering cause by occlusion or people opening the car's door and so on
            for fg_track in fg_objs:
                for bg_track in bg_objs:
                    if fg_track.track_id == bg_track.track_id:
                        # print(f'fg_leave_bg_id: {bg_track.track_id}')
                        bg_objs.remove(bg_track)
                        print(f'temp_miss_id: {bg_track.track_id}')

            # Determine the bg objs is still parking or not
            for bg_track in bg_objs:
                if bg_track.is_leave == True:
                    # print(f'is_leave_bg_id: {bg_track.track_id}')
                    bg_objs.remove(bg_track)
                    print(f'is_leave_id: {bg_track.track_id}')


            # bg_id = []
            # for bg_track in bg_objs:
            #     bg_id.append(bg_track.track_id)
            # print(f'bg_id: {bg_id}')


            # Output the final results
            bg_outputs = []
            fg_outputs = []
            for track in bg_objs:
                # if not track.is_confirmed() or track.time_since_update > 1:
                #     continue
                box = track.to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                bg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
            if len(bg_outputs) > 0:
                bg_outputs = np.stack(bg_outputs,axis=0)

            for track in fg_objs:
                # if not track.is_confirmed() or track.time_since_update > 1:
                #     continue
                box = track.to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                fg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
            if len(fg_outputs) > 0:
                fg_outputs = np.stack(fg_outputs,axis=0)

            # Put the result into queue to return
            queue.put(bg_outputs)
            queue.put(fg_outputs)
        """
        ##############################################################

        # Ver.3-1
        ##############################################################
        # background tracking results
        if cam_index == 0:
            queue.put(self.tracker.tracks[0])
        '''
        <DEFINITIONS>
        foreground objects: The bboxes of cars that are moving in the current frame
        background objects: The bboxes of cars that are parked well in the current frame 
                            (1) The bboxes of parked cars in bg_tracks
                            (2) The bboxes that is moving because of occlusion in ori_tracks
        TODO:
            1. Determine the obj in original frame is fg/bg obj
            2. Deal with the case that a car is parking in
            3. Deal with the case that a car is leaving the parking lot
        '''
        
        if cam_index == 1:
            # The real background/foreground objects that are going to be plot in the video
            bg_objs = []
            fg_objs = []

            # the background objects in orginal frame (those we don't want to plot)
            InOri_bg_objs = []
            InOri_temp_miss = []
            InOri_noise_objs = []
            Inbg_noise_objs = []

            # The car that already parked well
            parked = []

            # The car that its foreground has completely left the parking lot
            bg_is_leave = []

            # Get the tracks informations in background frame
            self.tracker.tracks[0] = queue.get()

            # fg_id_initial = []
            # for fg_track in self.tracker.tracks[1]:
            #     fg_id_initial.append(fg_track.track_id)
            # print(f'fg_id_initial: {fg_id_initial}')
            # st=time.time()
            # Traverse through all bboxes of background frame
            for bg_indice, bg_track in enumerate(self.tracker.tracks[0]):

                # For 3.
                # To find biggest bg_overlap, if < 0.35, means there is no ori_bbox overlap with this bg_bbox, so this car is already left
                max_overlap = 0.0
                leave = None

                # Traverse through all bboxes of original frame
                for ori_indice, ori_track in enumerate(self.tracker.tracks[1]):
                    # Transform the bbox into: (x1, y1, x2, y2)
                    box = ori_track.to_tlwh()
                    ori_bb = self._tlwh_to_xyxy(box)
                    box = bg_track.to_tlwh()
                    bg_bb = self._tlwh_to_xyxy(box)
                    
                    # Calculate the top-left and bottom-right coordinaries of two bbox
                    ori_bb1 = [ori_bb[0], ori_bb[1]] # origin obj bbox top-left (x1, y2)
                    ori_bb2 = [ori_bb[2], ori_bb[3]] # origin obj bbox bottom-right (x2, y2)
                    bg_bb1 = [bg_bb[0], bg_bb[1]]    # background obj bbox top-left (x1, x2)
                    bg_bb2 = [bg_bb[2], bg_bb[3]]    # background obj bbox bottom-right (x2, y2)
                    

                    bg_bb_area = (bg_bb2[0] - bg_bb1[0]) * (bg_bb2[1] - bg_bb1[1])
                    ori_bb_area = (ori_bb2[0] - ori_bb1[0]) * (ori_bb2[1] - ori_bb1[1])

                    if ori_bb_area <= 3000:
                        InOri_noise_objs.append(ori_indice)
                    
                    if bg_bb_area <= 3000:
                        Inbg_noise_objs.append(bg_indice)
                    
                    # determine the coordinates of the intersection rectangle
                    x_left = max(ori_bb1[0], bg_bb1[0])
                    y_top = max(ori_bb1[1], bg_bb1[1])
                    x_right = min(ori_bb2[0], bg_bb2[0])
                    y_bottom = min(ori_bb2[1], bg_bb2[1])
                    

                    # In case that there is no intersection
                    if x_right < x_left or y_bottom < y_top:
                        overlap_ori = 0.0
                        overlap_bg = 0.0

                    else:
                        # Calculate intersection
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)

                        # Calculate overlap of (1)bg_bb_area and intersection_area (2)fg_bbox_area and intersection_area
                        # bg_bb_area = (bg_bb2[0] - bg_bb1[0]) * (bg_bb2[1] - bg_bb1[1])
                        # ori_bb_area = (ori_bb2[0] - ori_bb1[0]) * (ori_bb2[1] - ori_bb1[1])

                        if bg_bb_area == 0:
                            overlap_bg = 0
                        else:
                            overlap_bg = intersection_area / float(bg_bb_area)
                        if ori_bb_area == 0:
                            overlap_ori = 0
                        else:
                            overlap_ori = intersection_area / float(ori_bb_area)


                    if not ori_track.is_confirmed() or ori_track.time_since_update > 1:
                        InOri_temp_miss.append(ori_indice)

                    # For 1.
                    if overlap_ori > 0.8:
                        InOri_bg_objs.append(ori_indice)
                        # InOri_bg_obj_ID.append(ori_track.track_id)

                    # for 2.
                    if overlap_ori > 0.8 and overlap_bg > 0.8:
                        parked.append([ori_indice, bg_indice])

                    # for 3.
                    if overlap_bg > max_overlap:
                        max_overlap = overlap_bg
                        leave = bg_indice


                    # if ori_track.track_id == 1:
                    #     print(f'1_overlap_ori_BEF: {overlap_ori}')


                # For 3.
                # If the max_overlap is < 0.35 means the car of this bg is left
                if max_overlap != 0 and max_overlap < 0.35:
                    bg_is_leave.append(leave)

            # print('[Time] Loop1 time: ', time.time() - st)
            ## 2. Deal with the case that a car is parking in
            if len(parked) != 0:
                for pair in parked:
                    ori_index = pair[0]
                    bg_index = pair[1]
                    
                    # When the bg_box just appear
                    if self.tracker.tracks[0][bg_index].is_parked is False:
                        # Only change at the first time
                        self.tracker.tracks[0][bg_index].is_parked = True
                        # Give the foreground id to the just appear background bbox
                        new_id = self.tracker.tracks[1][ori_index].track_id
                        old_id = self.tracker.tracks[0][bg_index].track_id
                        # Give the foreground id to the just appear background bbox
                        self.tracker.tracks[0][bg_index].track_id = self.tracker.tracks[1][ori_index].track_id
                        # Change the samples in tracker.metric too!!!!!
                        if self.tracker.metric[0].samples.get(old_id) != None:
                            self.tracker.metric[0].samples[new_id] = self.tracker.metric[0].samples.pop(old_id)

                    # Always keep the parked car's fg_id is the same with its bg_id 
                    if self.tracker.tracks[0][bg_index].is_parked is True:
                        new_id = self.tracker.tracks[0][bg_index].track_id
                        old_id = self.tracker.tracks[1][ori_index].track_id

                        # if new_id == 1:
                            # print(f'ori_id: {old_id}')
                            # print(f'bg_id: {new_id}')

                        # Give the background id to its foreground
                        self.tracker.tracks[1][ori_index].track_id = self.tracker.tracks[0][bg_index].track_id
                        # Change the samples in tracker.metric too!!!!!
                        if self.tracker.metric[1].samples.get(old_id) != None:
                            self.tracker.metric[1].samples[new_id] = self.tracker.metric[1].samples.pop(old_id)

            ## 3. Deal with the case that a car is totally left the parking lot, none of the ori_bbox is overlap with the bg_bbox exceed 0.35
            if len(bg_is_leave) != 0:
                for bg_index in bg_is_leave:
                    if self.tracker.tracks[0][bg_index].is_leave is False:
                        # Only change at the first time
                        self.tracker.tracks[0][bg_index].is_leave = True

            
            # fg_id_bef_rmbg = []
            # for fg_track in self.tracker.tracks[1]:
            #     fg_id_bef_rmbg.append(fg_track.track_id)
            # print(f'fg_id_bef_rmbg: {fg_id_bef_rmbg}')

            
            ## 1. Determine the objs in original frame is a fg or bg objs
            for ori_indice, ori_track in enumerate(self.tracker.tracks[1]):
                if (ori_indice not in InOri_bg_objs) and (ori_indice not in InOri_temp_miss): #and (ori_indice not in InOri_noise_objs)
                    fg_objs.append(ori_track)


            # print(f'InOri_bg_objs: {InOri_bg_objs}')
            # print(f'InOri_bg_obj_ID: {InOri_bg_obj_ID}')
            
            # fg_id_final = []
            # for fg_track in fg_objs:
            #     fg_id_final.append(fg_track.track_id)
            # print(f'fg_id_final: {fg_id_final}')

            
            # Put all bg_track into bg_obj (we will decide witch to plot later)
            for bg_indice, bg_track in enumerate(self.tracker.tracks[0]):
                # if bg_indice not in Inbg_noise_objs:
                #     bg_objs.append(bg_track)
                bg_objs.append(bg_track)


            # If a car is in fg_objs (going to plot), and its background is exist, then plot the fg_obj, there is two cases:
            # (1) The parked car is leaving, but not totally leave the scene yet
            # (2) The original bbox jittering cause by occlusion or people opening the car's door and so on
            for fg_track in fg_objs:
                for bg_track in bg_objs:
                    if fg_track.track_id == bg_track.track_id:
                        # print(f'fg_leave_bg_id: {bg_track.track_id}')
                        bg_objs.remove(bg_track)

            # Determine the bg objs is still parking or not
            for bg_track in bg_objs:
                if bg_track.is_leave == True:
                    # print(f'is_leave_bg_id: {bg_track.track_id}')
                    bg_objs.remove(bg_track)


            # Output the MOT tracker
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            # st=time.time()
            out_path = "/mnt/HDD1/jinghung/YOLOX_deepsort_tracker_mulThread/MOT_eval/03.txt"
            for track in bg_objs:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                line = [str(frame), str(track.track_id), str('%.6f'%box[0]), str('%.6f'%box[1]), str('%.6f'%box[2]), str('%.6f'%box[3]), str('%.6f'%track.conf), str(-1), str(-1), str(-1)]
                line = ','.join(line) + '\n'
                with open(out_path , 'a') as f:
                    f.writelines(line) 
            
            for track in fg_objs:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                line = [str(frame), str(track.track_id), str('%.6f'%box[0]), str('%.6f'%box[1]), str('%.6f'%box[2]), str('%.6f'%box[3]), str('%.6f'%track.conf), str(-1), str(-1), str(-1)]
                line = ','.join(line) + '\n'
                with open(out_path , 'a') as f:
                    f.writelines(line) 
            # print('[Time] IO time', time.time() - st)

            # Output the final results
            bg_outputs = []
            fg_outputs = []
            for track in bg_objs:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                bg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
            if len(bg_outputs) > 0:
                bg_outputs = np.stack(bg_outputs,axis=0)

            for track in fg_objs:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
                track_id = track.track_id
                fg_outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
            if len(fg_outputs) > 0:
                fg_outputs = np.stack(fg_outputs,axis=0)

            # Put the result into queue to return
            queue.put(bg_outputs)
            queue.put(fg_outputs)
        ##############################################################

        '''
        algorithm1 Fusion
        Input: background frame tracks BG = {track_1,...,track_n}, 
            original frame tracks ORI = {track_1,...,track_n}

        for bg_track in BG do
            for ori_track in ORI do
                Compute overlap_bg using Eq.?
                Compute overlap_ori using Eq.?
                Construct InOri_bg_obj
                Construct parked
            end for
        end for        
        fg_objs <- {ori_track 屬於 ORI | ori_track 不屬於 InOri_bg_obj}
        bg_objs <- {bg_track 屬於 BG}
        output <- fg_obls + bg_objs
        return output
        '''
        

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


