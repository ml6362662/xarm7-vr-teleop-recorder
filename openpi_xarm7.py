"""OpenPI adapter for the default absolute-joint LeRobot v2.1 export.

Copy/import into your chosen OpenPI checkout. This file does not control a robot.
Training factory imports OpenPI lazily; array transforms can be tested alone.
"""
import dataclasses

import numpy as np


def rgb_uint8(value):
    image=np.asarray(value)
    if image.ndim!=3:
        raise ValueError("expected a 3-dimensional RGB image")
    if image.shape[0]==3 and image.shape[-1]!=3:
        image=np.moveaxis(image,0,-1)
    if image.shape[-1]!=3:
        raise ValueError("expected RGB channels")
    if np.issubdtype(image.dtype,np.floating):
        if not np.isfinite(image).all() or image.min()<0 or image.max()>1:
            raise ValueError("floating point images must be RGB in [0,1]")
        image=np.rint(image*255).astype(np.uint8)
    if image.dtype!=np.uint8:
        raise ValueError("expected uint8 RGB or normalized float RGB")
    return image


@dataclasses.dataclass(frozen=True)
class XArm7Inputs:
    def __call__(self,data):
        state=np.asarray(data["state"],dtype=np.float32)
        if state.shape!=(8,) or not np.isfinite(state).all():
            raise ValueError("xArm7 state must be 7 joints in rad + open fraction")
        base=rgb_uint8(data["images"]["cam_high"])
        wrist=rgb_uint8(data["images"]["cam_left_wrist"])
        result=dict(state=state,image={"base_0_rgb":base,"left_wrist_0_rgb":wrist,
                    "right_wrist_0_rgb":np.zeros_like(wrist)},
                    image_mask={"base_0_rgb":np.True_,"left_wrist_0_rgb":np.True_,"right_wrist_0_rgb":np.False_})
        if "actions" in data:
            actions=np.asarray(data["actions"],dtype=np.float32)
            if actions.ndim!=2 or actions.shape[-1]!=8 or not np.isfinite(actions).all():
                raise ValueError("expected [horizon,8] absolute joint/gripper commands")
            result["actions"]=actions
        if "prompt" in data:
            result["prompt"]=data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class XArm7Outputs:
    def __call__(self,data):
        # Padding is an OpenPI model transform; it is never saved as fake joints.
        return {"actions":np.asarray(data["actions"])[...,:8]}


def make_training_config(repo_id,*,batch_size=8,action_horizon=30):
    """Create a registration-ready config for pi05_base, default absolute export.

    Recompute normalization on this dataset; do not use ALOHA/DROID robot stats.
    Native joint actions become deltas only in the training transform and are
    reconstructed to absolute joint targets by the paired output transform.
    """
    from openpi import transforms as tr
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as cfg
    from openpi.training import weight_loaders

    @dataclasses.dataclass(frozen=True)
    class XArm7DataConfig(cfg.DataConfigFactory):
        def create(self,assets_dirs,model_config):
            delta_mask=tr.make_bool_mask(7,-1)
            data_transforms=tr.Group(inputs=[XArm7Inputs()],outputs=[XArm7Outputs()])
            data_transforms=data_transforms.push(inputs=[tr.DeltaActions(delta_mask)],outputs=[tr.AbsoluteActions(delta_mask)])
            return dataclasses.replace(self.create_base_config(assets_dirs,model_config),
                repack_transforms=tr.Group(inputs=[tr.RepackTransform({
                    "images":{"cam_high":"observation.images.cam_high","cam_left_wrist":"observation.images.cam_left_wrist"},
                    "state":"observation.state","actions":"action"})]),
                data_transforms=data_transforms,model_transforms=cfg.ModelTransformFactory()(model_config),
                action_sequence_keys=("action",),prompt_from_task=True)

    return cfg.TrainConfig(name="pi05_xarm7",model=Pi0Config(pi05=True,action_dim=32,action_horizon=action_horizon),
        data=XArm7DataConfig(repo_id=repo_id),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=batch_size,num_train_steps=30000)
