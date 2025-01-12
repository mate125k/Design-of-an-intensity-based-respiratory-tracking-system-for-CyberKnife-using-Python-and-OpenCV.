import cv2 as cv
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import csv


# Mentéshez 
directory_path = 'C:\\7_felev\\szakdolgozat_meres\\intenzitas\\1_meres_png'
valami_dir = 'C:\\7_felev\\szakdolgozat_meres\\intenzitas\\1_meres_gray'  # ellenőrzésre
valami_dir2 = 'C:\\7_felev\\szakdolgozat_meres\\intenzitas\\1_meres_binary' #bináris képek tresholdot alkalmazva  
cropped_dir = 'C:\\7_felev\\szakdolgozat_meres\\intenzitas\\1_int_cropped_image' #Roi elmentjük a kivágott részeleteket 
test_dir= 'C:\\7_felev\\szakdolgozat_meres\\intenzitas\\test'
dok_dir= 'C:\\7_felev\\szakdolgozat_meres\\intenzitas\\dokumentalas' # Beolvassuk a a képeket 


images = glob.glob(f'{directory_path}/*.png')
gray_images = glob.glob(f'{valami_dir}/*.png')
binary_images= glob.glob(f'{valami_dir2}/*.png')
cropped_images = glob.glob(f'{cropped_dir}/*.png')
frames =sorted(glob.glob(f'{dok_dir}/*.png'))

###### Function #######

def display_resized_image(img, window_name, max_width=800, max_height=600): 
    """Resize and display an image"""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1)  
    if scale < 1:
        img = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    cv.imshow(window_name, img)
    
    cv.destroyWindow(window_name)  
    return img

def preprocess_image(gray_image):
    #Trehsolding 
    _, binary_image = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY_INV)

    # Megnézzük a szürkeárnyalaots hisztogrammon lévő pontok kontrsztjai a kép melyik részéhez tartoznak 

    #binary_image1=display_resized_image(binary_image, "first image")
    #cv.imshow("Binary Image", binary_image1)
    # Ablak nyitva tartása, amíg egy billentyűt le nem nyomunk
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return binary_image

def select_roi(image):
    # ROI kiválasztása 
    roi = cv.selectROI("Select ROI", image, fromCenter=False, showCrosshair=False)
    cv.destroyWindow("Select ROI")
    return roi 

def detect_features(image, max_corners=100):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Detecting good features to track using Shi-Tomasi method
    corners = cv.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    
    if corners is not None:
        intcorners = np.int0(corners)  
        # Rajzoljuk körbe a pontokat amiket kovetni fogunk
    for i in intcorners:
        x, y = i.ravel()
        cv.circle(image, (x, y), 2, (0, 255, 0), 1)


       

     # Minden egyes pont amit megtaláltunk 
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    return corners

def track_dots_with_optical_flow(images):
    
    first_image = cv.imread(images[0])

   
    prev_points = detect_features(first_image)

    
  
    print("Initial positions of the dots in the first frame:")


    for p in prev_points:
        
        print(f"Dot position: {p.ravel()}")
        dot_position = " ".join([f"{coord:.2f}" for coord in p.ravel()])
    
    #CSV fájl
    output_csv_path = os.path.join(test_dir, "displacement_data.csv")
    csv_file = open(output_csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Old_X", "Old_Y", "New_X", "New_Y", "Displacement_X", "Displacement_Y"])
   

    for i, image_path in enumerate(images[1:], start=1):
        current_image = cv.imread(image_path)
      

        # Optical flow
        next_points, status, error = cv.calcOpticalFlowPyrLK(first_image, current_image, prev_points, None)

        # Csak azokat nézzük amik lekövethetők voltak 
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        

        print(f"\nDisplacement of dots in Frame {i} (relative to the first frame):")


        
        for old, new in zip(good_old, good_new):
            old_x, old_y = old.ravel()
            new_x, new_y = new.ravel()
            

            # elmozdulás meghatározása 
            displacement_x = new_x - old_x
            displacement_y = new_y - old_y

            
            print(f"Old position: ({old_x:.2f}, {old_y:.2f}), New position: ({new_x:.2f}, {new_y:.2f}), "
                  f"Displacement: ({displacement_x:.2f}, {displacement_y:.2f})")


            
            # CSV fájl kiirjuk az adatokat 
            csv_writer.writerow([i, old_x, old_y, new_x, new_y, displacement_x, displacement_y]) 
        
               

      
               
                



           
            # kössük ossze elmozdult pontok helyzetetét 
            cv.circle(current_image, (int(new_x), int(new_y)), 2, (0, 0, 255), 1)  # Draw red circle
            cv.line(current_image, (int(old_x), int(old_y)), (int(new_x), int(new_y)), (255, 0, 0), 2)

       
        cv.imshow(f"Optical Flow - Frame {i}", current_image)
        output_filename = os.path.join(test_dir, f"test_image_{i}.png")
        cv.imwrite(output_filename, current_image)  
        cv.destroyAllWindows()

    csv_file.close()

def create_video(frames, output_path, fps=10):
    first_frame = cv.imread(frames[0])
    height, width, _ = first_frame.shape  # első kép
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Mindent a videóba 
    for frame_path in frames:
        frame = cv.imread(frame_path)  
        out.write(frame)  
    
    out.release()

# Inicializálás ##
first_dots = None  # A etalon képhez lévő pontok 
roi = None  # A kiejölt területe az érdekelt területnek 

for i, image in enumerate(images):  # Fekete fehérbe 
    print(images)

    img = cv.imread(image)
    print(image)

    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Normalizálás, hogy a képadatok 0 és 1 között legyenek
    gray_image = gray_image.astype(np.float32) / 255.0

    # Kép hisztogramjának kiszámítása
    histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0.0, 1.0))

    # Ábrázolás
    #fig, ax = plt.subplots()
    #ax.plot(bin_edges[:-1], histogram)  # Az utolsó bin-et nem használjuk
    #ax.set_title("Normalizált szürkeárnyalatos Hisztogram")
    #ax.set_xlabel("szürke árnyalatos érték")
    #ax.set_ylabel("Pixel szám")
    #ax.set_xlim(0, 1.0)

    output_filename = os.path.join(valami_dir, f"gray_image_3_measue_int{i}.png")
    cv.imwrite(output_filename, gray_image)


for i, gray_image in enumerate(gray_images):    # treshold objektum 
    
    gray = cv.imread(gray_image)
    binary_image = preprocess_image(gray)
    output_filename = os.path.join(valami_dir2, f"binary_image_3_measue_int{i}.png")
    cv.imwrite(output_filename, binary_image)


for i, binary_image in enumerate(binary_images): #ROI kivágása 

    binary = cv.imread(binary_image)
   

    if i==0 and roi is None:
        # Hogy teljes képben lássuk a képet 
        window_binary_image = display_resized_image(binary, "first image")
        roi = select_roi(window_binary_image)  # válasszuk ki az érdekelt területet 

    window_binary_image = display_resized_image(binary, "All te other")
    #  Vágjuk ki a képből
    cropped_image = window_binary_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    

    cv.imshow(f"Cropped image {i}", cropped_image)
    output_filename = os.path.join(cropped_dir, f"cropped_image_{i}.png")
    cv.imwrite(output_filename, cropped_image)  
    cv.destroyAllWindows()  


track_dots_with_optical_flow(cropped_images)

output_path = os.path.join(dok_dir, f"video.mp4")
create_video(frames, output_path, fps=1)




    
    












    

   


    





