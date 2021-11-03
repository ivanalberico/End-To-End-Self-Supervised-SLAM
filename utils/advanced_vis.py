import  torch
import plotly
import gradslam as gs
import numpy as np
import plotly.graph_objects as go

def plotly_map_update_visualization(intermediate_pcs, poses, K, max_points_per_pc=50000, ms_per_frame=50):
    """
    Args:
        - intermediate_pcs (List[gradslam.Pointclouds]): list of gradslam.Pointclouds objects, each of batch size 1
        - poses (torch.Tensor): poses for drawing frustums
        - K (torch.Tensor): Intrinsics matrix
        - max_points_per_pc (int): maximum number of points to plot for each pointcloud
        - ms_per_frame (int): miliseconds per frame for the animation

    Shape:
        - poses: :math:`(L, 4, 4)`
        - K: :math:`(4, 4)`
    """
    def plotly_poses(poses, K):
        """
        Args:
            poses (np.ndarray):
            K (np.ndarray):

        Shapes:
            - poses: :math:`(L, 4, 4)`
            - K: :math:`(4, 4)`
        """
        fx = abs(K[0, 0])
        fy = abs(K[1, 1])
        f = (fx + fy) / 2
        cx = K[0, 2]
        cy = K[1, 2]

        cx = cx / f
        cy = cy / f
        f = 1.

        pos_0 = np.array([0., 0., 0.])
        fustum_0 = np.array(
            [
                [-cx, -cy, f],
                [cx, -cy, f],
                list(pos_0),
                [-cx, -cy, f],
                [-cx, cy, f],
                list(pos_0),
                [cx, cy, f],
                [-cx, cy, f],
                [cx, cy, f],
                [cx, -cy, f],
            ]
        )

        traj = []
        traj_frustums = []
        for pose in poses:
            rot = pose[:3, :3]
            tvec = pose[:3, 3]

            fustum_i = fustum_0 @ rot.T
            fustum_i = fustum_i + tvec
            pos_i = pos_0 + tvec

            pos_i = np.round(pos_i, decimals=2)
            fustum_i = np.round(fustum_i, decimals=2)

            traj.append(pos_i)
            traj_array = np.array(traj)
            traj_frustum = [
                go.Scatter3d(
                    x=fustum_i[:, 0], y=fustum_i[:, 1], z=fustum_i[:, 2],
                    marker=dict(
                        size=0.1,
                    ),
                    line=dict(
                        color='purple',
                        width=4,
                    )
                ),
                go.Scatter3d(
                    x=pos_i[None, 0], y=pos_i[None, 1], z=pos_i[None, 2],
                    marker=dict(
                        size=6.,
                        color='purple',
                    )
                ),
                go.Scatter3d(
                    x=traj_array[:, 0], y=traj_array[:, 1], z=traj_array[:, 2],
                    marker=dict(
                        size=0.1,
                    ),
                    line=dict(
                        color='purple',
                        width=2,
                    )
                ),
            ]
            traj_frustums.append(traj_frustum)
        return traj_frustums

    def frame_args(duration):
        return {
            "frame": {"duration": duration, "redraw": True},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    # visualization
    scatter3d_list = [pc.plotly(0, as_figure=False, max_num_points=max_points_per_pc) for pc in intermediate_pcs]
    traj_frustums = plotly_poses(poses.cpu().numpy(), K.cpu().numpy())
    data = [[*frustum, scatter3d] for frustum, scatter3d in zip(traj_frustums, scatter3d_list)]

    steps = [{"args": [[i], frame_args(0)], "label": i, "method": "animate"} for i in range(len(intermediate_pcs))] # TODO: SEQ LEN PARAM Hardcoded
    sliders = [
        {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {"prefix": "Frame: "},
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": steps,
        }
    ]
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [None, frame_args(ms_per_frame)],
                    "label": "&#9654;",
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "&#9724;",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    fig = go.Figure()
    frames = [{"data": frame, "name": i} for i, frame in enumerate(data)]
    fig.add_traces(frames[0]["data"])
    fig.update(frames=frames)
    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=False,
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False,),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False,),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False,),
        )
    )
    fig.show()
    return fig
