from utils.visualization import generate_degradation_plots

if __name__ == "__main__":
    directories = [
            "./ViT_Benchmark_Results",
            "./MicroViT_Benchmark_Results",
            "./ResNet_Benchmark_Results",
            "./MobileNet_Benchmark_Results"
    ]

    full_df = generate_degradation_plots(dir_paths = directories, csv_filename = "benchmark_summary.csv")
