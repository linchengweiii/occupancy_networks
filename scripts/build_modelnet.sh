MESHFUSION_PATH=/home/cwlin/research/dataset_generation/external/mesh-fusion
INPUT_PATH=/home/cwlin/datasets/ModelNet40
BUILD_PATH=/home/cwlin/ModelNet40.build

CLASSES=(
airplane
bathtub
bed
bench
bookshelf
bottle
bowl
car
chair
cone
cup
curtain
desk
door
dresser
flower_pot
glass_box
guitar
keyboard
lamp
laptop
mantel
moniter
night_stand
person
piano
plant
radio
range_hood
sink
sofa
stairs
stool
table
tent
toilet
tv_stand
vase
wardrobe
xbox
)

mkdir -p $BUILD_PATH

for class in ${CLASSES[@]}; do
    echo "Parsing class $class"
    for split in train test; do
        echo "Split $split"
        INPUT_PATH_CLASS=$INPUT_PATH/$class/$split
        BUILD_PATH_CLASS=$BUILD_PATH/$class/$split

        mkdir -p $BUILD_PATH_CLASS/1_scaled \
                 $BUILD_PATH_CLASS/1_transfrom \
                 $BUILD_PATH_CLASS/2_depth \
                 $BUILD_PATH_CLASS/2_watertight \
                 $BUILD_PATH_CLASS/4_points \
                 $BUILD_PATH_CLASS/4_pointcloud \
                 $BUILD_PATH_CLASS/4_watertight_scaled \

        echo "Scaling meshes"
        python $MESHFUSION_PATH/1_scale.py \
          --in_dir $INPUT_PATH_CLASS \
          --out_dir $BUILD_PATH_CLASS/1_scaled \
          --t_dir $BUILD_PATH_CLASS/1_transform

        echo "Create depths maps"
        python $MESHFUSION_PATH/2_fusion.py \
          --mode render \
          --in_dir $BUILD_PATH_CLASS/1_scaled \
          --out_dir $BUILD_PATH_CLASS/2_depth

        echo "Produce watertight meshes"
        python $MESHFUSION_PATH/2_fusion.py \
          --mode fuse \
          --in_dir $BUILD_PATH_CLASS/2_depth \
          --out_dir $BUILD_PATH_CLASS/2_watertight \
          --t_dir $BUILD_PATH_CLASS/1_transform

        echo "Process watertight meshes"
        python sample_mesh.py $BUILD_PATH_CLASS/2_watertight \
            --resize \
            --bbox_in_folder $INPUT_PATH_CLASS \
            --pointcloud_folder $BUILD_PATH_CLASS/4_pointcloud \
            --points_folder $BUILD_PATH_CLASS/4_points \
            --mesh_folder $BUILD_PATH_CLASS/4_watertight_scaled \
            --packbits --float16

    done
done
