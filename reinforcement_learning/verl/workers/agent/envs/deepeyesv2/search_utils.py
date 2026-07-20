import json
import os
import random
import time
import warnings
from pathlib import Path

from PIL import Image

_repo_root = Path(__file__).resolve().parents[6]
_default_cache_path = _repo_root / 'data/search_cache/fvqa_train_image_search_results_cache.json'
_configured_cache_paths = os.getenv('DEEPEYES_SEARCH_CACHE_PATHS')
mmsearch_r1_cache_json_path_list = (
    [Path(path).expanduser() for path in _configured_cache_paths.split(os.pathsep) if path]
    if _configured_cache_paths
    else [_default_cache_path]
)

mmsearch_r1_cache_json = {}
for cache_json_path in mmsearch_r1_cache_json_path_list:
    if not cache_json_path.is_file():
        continue
    try:
        with cache_json_path.open('r', encoding='utf-8') as f:
            mmsearch_r1_cache_json.update(json.load(f))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f'Failed to load DeepEyes search cache {cache_json_path}: {exc}', stacklevel=1)

# We just show the search function here, which is a placeholder.
# You can replace it with your actual search implementation.
def search(query, size=5):
    max_try = 3
    
    result = 'Error'
    for try_idx in range(max_try):
        client = None
        try:

            result = {'elapsed_time': 0.0, 'data': []}
            for i in range(5):
                search_info = {}
                search_info['snippet'] = "This is a placeholder snippet for query: " + query
                search_info['title'] = "Placeholder Title " + str(i)
                search_info['link'] = "http://example.com/" + str(i)
                result['data'].append(search_info)

            break
        except Exception as e:
            print (e.message.decode('utf-8'))
            result = 'Error'
            if try_idx < max_try - 1:
                sleep_time = (try_idx + 1) * random.randint(1, 5)
                time.sleep(sleep_time)
        finally:
            pass
    if try_idx == (max_try-1):
        print('Failed to search', query)

    return result

def image_search(query, data_idx=None):
    if data_idx and 'fvqa' in str(data_idx):
        cached_data = mmsearch_r1_cache_json.get(data_idx, {})
        if not cached_data:
            return 'Error'
        tool_returned_web_title = cached_data['tool_returned_web_title']
        cached_images_path = cached_data['cached_images_path']
        return_cached_images_path, return_tool_returned_web_title = [], []
        for i in range(len(cached_images_path)):
            if cached_images_path[i]is not None and os.path.exists(cached_images_path[i]):
                return_cached_images_path.append(cached_images_path[i])
                return_tool_returned_web_title.append(tool_returned_web_title[i])

        return {
            "tool_returned_web_title": return_tool_returned_web_title,
            "cached_images_path": return_cached_images_path
        }
    else:
        return 'Error'
