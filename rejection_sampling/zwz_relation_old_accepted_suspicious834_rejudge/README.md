# Old accepted relation-query rejudge

This batch contains the 834 trajectories from the earlier 8,755 accepted rows whose `grounding_detect.query` or `crop_zoom.label` contains a spatial-relation phrase.

Only the original image is sent to the judge. The full recorded tool trajectory, detector boxes, crop coordinates, final answer, and suspicious tool fields remain in the request. The mixed-task V4.1 prompt treats answer-bearing or answer-presupposing tool queries as a hard rejection.

The job uses the official DeepSeek endpoint with 60 RPM, eight workers, retries, and resumable output.
