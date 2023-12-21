Download shift using the official download script [download.py](https://raw.githubusercontent.com/SysCV/shift-dev/main/download.py) with the following options:

```shell
python download.py --view  "[front]" \       # list of view abbreviation to download
                   --group "[img, det2d]" \  # list of data group abbreviation to download 
                   --split "[train, val]" \  # list of splits to download 
                   --framerate "[videos]" \  # chooses the desired frame rate (images=1fps, videos=10fps)
                   ./data                    # path where to store the downloaded data
```

More detailed instructions on the accepted arguments can be found at [https://www.vis.xyz/shift/download/](https://www.vis.xyz/shift/download/).