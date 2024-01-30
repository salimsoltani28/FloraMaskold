#Solutions for object image polygone detections
#1- we can make use of EBImage package to generate polygones

library(reticulate)
reticulate::use_condaenv("tfr", required = TRUE)
#Libraries
require(keras)
library(tensorflow)
library(tfdatasets)
library(tidyverse)
library(tibble)
library(magick)
library(viridis)
library(magrittr)
library(raster)
library(magick)
#set a seed
tf$random$set_seed(25L)
set.seed(25L)
# Disable eager execution
tf$compat$v1$disable_eager_execution()
# set memory growth policy
gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
#gpu2 <- tf$config$experimental$get_visible_devices('GPU')[[2]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)

#set your working directory
setwd("/scratch1/ssoltani/workshop/11_FloraMask/")
mask_folder <- "masks/"
#give the name of model dri 
model_name <- "Output_effnet7_stamfiltering_distover0.2_Under15m_img512_11classGOOGLEimg_2Dense_256_512/"

checkpoint_dir <-  paste0("model/", model_name)

# List all files in the directory that match the pattern
model_files <- list.files(checkpoint_dir, pattern = "weights.*hdf5")
# Extract loss values from the filenames
loss_values <- sapply(model_files, function(file) {
  as.numeric(unlist(strsplit(unlist(strsplit(file, "-"))[2], ".hdf5"))[1])
})

# Get the filename of the model with the lowest loss
best_model_file <- model_files[which.min(loss_values)]

#load the model
model = load_model_hdf5(paste0(checkpoint_dir, best_model_file), compile = FALSE)
summary(model)

#check which model is loaded
print(best_model_file)


files <- "dataset/1_example_photos_iNat/"

#Choose some images
test_images <- paste(files, list.files(files, recursive = T), sep = "")


set.seed(1)
sample_test_images = test_images
sample_test_images


#dataset load function 



load_and_process = function(img) {
  # img_out = tf$image$decode_jpeg(tf$io$read_file(img)
  #                            , channels = 3
  #                            #, ratio = down_ratio
  #                            , try_recover_truncated = TRUE
  #                            , acceptable_fraction=0.5
  # ) %>%
  #   tf$cast(dtype = tf$float32) %>%  
  #   tf$math$divide(255) %>% 
  #   #tf$image$convert_image_dtype(dtype = tf$float32) %>%
  #   tf$keras$preprocessing$image$smart_resize(size=c(512L, 512L)) %>% 
  #   image_to_array() %>%
  img_out = image_load(img, target_size = c(512, 512)) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, 512, 512, 3))
  return(img_out)
}




# Gradient-weighted Class Activation Mapping (Grad-CAM) function
grad_cam <- function(model, img_out, y, sample_test_images, mask_folder) {
  library(raster)
  library(magick)
  preds <- model %>% predict(img_out)
  max = which.max(preds[1,])
  test_image_output <- model$output[, max]
  last_conv_layer <- model %>% get_layer("top_conv") #last layer
  grads <- k_gradients(test_image_output, last_conv_layer$output)[[1]]
  pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
  iterate <- k_function(list(model$input),
                        list(pooled_grads, last_conv_layer$output[1,,,]))
  c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img_out))
  for (i in 1:512) {
    conv_layer_output_value[,,i] <-
      conv_layer_output_value[,,i] * pooled_grads_value[[i]]
  }
  
  # Average the weighted feature maps along the channels to create a heatmap
  heatmap <- apply(conv_layer_output_value, c(1, 2), mean)
  heatmap <- pmax(heatmap, 0)
  heatmap <- heatmap / max(heatmap)
  
  # Read the original image and get its dimensions
  image <- image_read(sample_test_images)
  info <- image_info(image)
  
  # Convert heatmap to a raster
  heatmap_raster <- image_read(as.raster(heatmap))
  # First, convert the image to grayscale to exclude color channels
  heatmap_gray <- image_convert(heatmap_raster, colorspace = 'gray')
  
  # Resize the raster to match the original image dimensions
  heatmap_resized <- image_resize(heatmap_gray, geometry = sprintf("%dx%d!", info$width, info$height), filter = "quadratic")
  image_write(heatmap_resized, path = paste(mask_folder,"img_heatmap_resized", y, ".jpg", sep = ""))
  # Convert the resized raster back to a matrix
  image_vector <- as.numeric(image_data(heatmap_resized))
  
  # Reshape the vector into a matrix with the correct dimensions
  # For a 500x333 image, ensure you have the correct number of elements in the vector
  heatmap_matrix <- matrix(image_vector, ncol =info$width, nrow = info$height, byrow = TRUE)
  
  ###
  write_heatmap <- function(heatmap, filename, width = 512, height = 512,
                            bg = "white", col = terrain.colors(12)) {
    png(filename, width = width, height = height, bg = bg)
    op = par(mar = c(0,0,0,0))
    on.exit({par(op); dev.off()}, add = TRUE)
    rotate <- function(x) t(apply(x, 2, rev))
    image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
  }
  
  #export the heatmap for visualization
  pal <- col2rgb(viridis(20), alpha = TRUE)
  #alpha <- floor(seq(0, 255, length = ncol(pal))) * 0.8
  alpha <- rep(128, ncol(pal))
  pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255) 
  write_heatmap(heatmap, paste("img_overlay_", y, ".png", sep = ""),
                width = 14, height = 14, bg = NA, col = pal_col)
}





for (y in 1:length(sample_test_images)) {
  image_temp <- load_and_process(sample_test_images[y])
  image_temp <- image_temp / 255
  geom_temp <- grad_cam(model, image_temp, y, sample_test_images[y],mask_folder = mask_folder,steps = 1)
  
}



#Plotting for visualization
# Plotting
#par(mfrow=c(5, 2), mar = c(rep(0.1, 4)))

for (y in 1:length(sample_test_images)) {
  image_temp <- load_and_process(sample_test_images[y])
  image_temp <- image_temp / 255
  geom_temp <- grad_cam(model, image_temp, y, sample_test_images[y],mask_folder = mask_folder)

  # Read the overlay image (heatmap)
  overlay_image <- image_read(paste("img_overlay_", y, ".png", sep = ""))
  overlay_image <- image_resize(overlay_image, geom_temp, filter = "quadratic")

  # Read the original image
  original_image <- image_read(sample_test_images[y])

  # Ensure the original image is in color
  original_image <- image_convert(original_image, colorspace = 'sRGB')

  # Composite the images
  #composite_image <- image_composite(original_image, overlay_image, operator = "atop")
  composite_image <- image_composite(original_image, overlay_image, operator = "atop")


  # Plot the composite image
  plot(composite_image)
}

