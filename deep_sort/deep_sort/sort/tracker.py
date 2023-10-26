# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from distutils.command.config import config
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .nn_matching import NearestNeighborDistanceMetric

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, max_dist, nn_budget, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = [ NearestNeighborDistanceMetric("euclidean", max_dist, nn_budget) for _ in range(2)]
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        print("max_age: ", max_age)
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = [[] for _ in range(2)]
        self._next_id = 1
        

    def predict(self, cam_index):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks[cam_index]:
            track.predict(self.kf)

    def update(self, detections, cam_index):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, cam_index)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[cam_index][track_idx].update(
                self.kf, detections[detection_idx])

        for track_idx in unmatched_tracks:
            self.tracks[cam_index][track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], cam_index)
        self.tracks[cam_index] = [t for t in self.tracks[cam_index] if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks[cam_index] if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks[cam_index]:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric[cam_index].partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections, cam_index):

        def gated_metric(tracks, dets, track_indices, detection_indices, cam_index):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
         
            # Get cost matrix
            cost_matrix = self.metric[cam_index].distance(features, targets)

            # Using gate matrix to constrain cost matrix
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            #print("with gate")

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks[cam_index]) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks[cam_index]) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric[cam_index].matching_threshold, self.max_age,
                self.tracks[cam_index], detections, cam_index, confirmed_tracks)    

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[cam_index][k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[cam_index][k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks[cam_index],
                detections, cam_index, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, cam_index):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks[cam_index].append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, detection.confidence, 
            detection.feature))
        self._next_id += 1
        if self._next_id > 1000000:
            self.next_id = 0