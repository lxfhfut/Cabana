# FAQs

**1. Can I analyze images with different pixel resolutions?**  
Yes, but it's not recommended. While Cabana can process images of varying resolutions, ridge detection results may vary significantly. For consistent and comparable analysis, use images with the same pixel resolution.

**2. What is the maximum supported image size?**  
Cabana supports images up to 2048×2048 pixels by default.

**3. Why was my image rejected due to a dark background?**  
Images with more than 99% of pixels having intensity values below 5 are automatically rejected. Such images are considered to lack sufficient regions of interest for analysis.

**4. How are images processed in Cabana?**  
Images are processed in batches of 5. This setup allows efficient processing and easier error recovery.

**5. What happens if Cabana crashes during processing?**  
If the program crashes before finishing, it can be restarted and will resume from the last successfully processed batch. Cabana automatically creates a checkpoint file to support this recovery.
