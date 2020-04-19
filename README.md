# fakecam
I just fetched a Open Source Virtual Background experiment from https://elder.dev/posts/open-source-virtual-background/ (props to [@BenTheElder](https://github.com/BenTheElder)) and tweaked it a bit:

* Added some scripts to ease running the experiment with Docker
* Added some more scripts to ease running the experiment **without** Docker (easier IMHO)
* Tweaked everything needed to make it run on the CPU instead of the GPU
  * Disabled GPU
  * Reduced the framerate
  * Reduced the resolution
* Improved the hologram effect adding a subtle animation to the scanlines and some of transparency
* Added more backgrounds, implemented video backgrounds, and made the backgrounds change automatically every minute
* Implemented command parameters to specify video devices, background paths, etc

## How to setup and run

* Without docker:
```
./run.sh
```
* With docker:
```
./run_docker.sh
```
