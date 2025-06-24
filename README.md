# CUDA Image Channel Separation

This project demonstrates GPU-accelerated channel separation (Red, Green, Blue) using CUDA and C++ with OpenCV.

## Project Structure

```
project-root/
├── data/               # Input images (place your .png/.jpg here)
├── processed/          # Output channel images
├── src/                # CUDA and C++ source code
├── headers/            # Header files
├── bin/                # Compiled binary
├── Makefile            # Build instructions
└── README.md           # Project documentation
```

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- OpenCV (with development headers)
- C++17 compatible compiler (e.g., g++)
- pkg-config

## Build Instructions

Open a terminal in the project root directory and run:

```bash
make
```

This will compile the sources and place the binary in `bin/channel_separation`.

## Running the Project

Ensure you have input images in the `data/` directory. Then run:

```bash
./bin/channel_separation [input_dir] [output_dir]
```

- `input_dir` (optional): Path to input images folder (default: `data`)
- `output_dir` (optional): Path for saving processed images (default: `processed`)

Example:

```bash
./bin/channel_separation data processed
```

The processed images will be saved in the specified output directory with filenames:

```
originalname-Red.png
originalname-Green.png
originalname-Blue.png
```

## Output

During execution, the program prints progress information and average channel values for each image.

## Clean

To remove compiled objects and binary, run:

```bash
make clean
```

## Automated Run Script

A convenience script (`run.sh`) automates cleaning, building, execution, and logging:

```bash
chmod +x run.sh      # make script executable (once)
./run.sh             # clean, build, run, and log to output.txt
```

The script will:
- Run `make clean` to remove all build and processed files
- Build the project (`make`)
- Execute `bin/channel_separation` on `data/` → `processed/`
- Log all console output to `output.txt`

## Author

**Shiven Saini**  
Email: shiven.career@proton.me
