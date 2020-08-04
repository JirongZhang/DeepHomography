import torch
import numpy as np
import cv2


def DLT_solve(src_p, off_set):
    # src_p: shape=(bs, n, 4, 2)
    # off_set: shape=(bs, n, 4, 2)
    # can be used to compute mesh points (multi-H)
   
    bs, _ = src_p.shape
    divide = int(np.sqrt(len(src_p[0])/2)-1)
    row_num = (divide+1)*2

    for i in range(divide):
        for j in range(divide):

            h4p = src_p[:,[2*j+row_num*i, 2*j+row_num*i+1, 
                    2*(j+1)+row_num*i, 2*(j+1)+row_num*i+1, 
                    2*(j+1)+row_num*i+row_num, 2*(j+1)+row_num*i+row_num+1,
                    2*j+row_num*i+row_num, 2*j+row_num*i+row_num+1]].reshape(bs, 1, 4, 2)  
            
            pred_h4p = off_set[:,[2*j+row_num*i, 2*j+row_num*i+1, 
                    2*(j+1)+row_num*i, 2*(j+1)+row_num*i+1, 
                    2*(j+1)+row_num*i+row_num, 2*(j+1)+row_num*i+row_num+1,
                    2*j+row_num*i+row_num, 2*j+row_num*i+row_num+1]].reshape(bs, 1, 4, 2)

            if i+j==0:
                src_ps = h4p
                off_sets = pred_h4p
            else:
                src_ps = torch.cat((src_ps, h4p), axis = 1)    
                off_sets = torch.cat((off_sets, pred_h4p), axis = 1)

    bs, n, h, w = src_ps.shape

    N = bs*n

    src_ps = src_ps.reshape(N, h, w)
    off_sets = off_sets.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = torch.ones(N, 4, 1)
    if torch.cuda.is_available():
        ones = ones.cuda()
    xy1 = torch.cat((src_ps, ones), 2)
    zeros = torch.zeros_like(xy1)
    if torch.cuda.is_available():
        zeros = zeros.cuda()

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1), 
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(N, 8)
 
    H = torch.cat((h8, ones[:,0,:]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    return H

 
def transformer(U, theta, out_size, **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):

        rep = torch.ones([n_repeats, ]).unsqueeze(0)
        rep = rep.int()
        x = x.int()

        x = torch.matmul(x.reshape([-1,1]), rep)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size, scale_h):

        num_batch, num_channels , height, width = im.size()

        height_f = height
        width_f = width
        out_height, out_width = out_size[0], out_size[1]

        zero = 0
        max_y = height - 1
        max_x = width - 1
        if scale_h:

            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = torch.from_numpy( np.array(width) )
        dim1 = torch.from_numpy( np.array(width * height) )

        base = _repeat(torch.arange(0,num_batch) * dim1, out_height * out_width)
        if torch.cuda.is_available():
            dim2 = dim2.cuda()
            dim1 = dim1.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()
            x0 = x0.cuda()
            x1 = x1.cuda()
            base = base.cuda()
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # channels dim
        im = im.permute(0,2,3,1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(height * width * num_batch,num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(height * width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(height * width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(height * width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa*Ia+wb*Ib+wc*Ic+wd*Id

        return output

    def _meshgrid(height, width, scale_h):

        if scale_h:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1), 1, 0))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                               torch.ones([1, width]))
        else:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(0.0, width.float(), width), 1), 1, 0))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height.float(), height), 1),
                               torch.ones([1, width]))


        x_t_flat = x_t.reshape((1, -1)).float()
        y_t_flat = y_t.reshape((1, -1)).float()

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0)
        if torch.cuda.is_available():
            grid = grid.cuda()
        return grid

    def _transform(theta, input_dim, out_size, scale_h):
        num_batch, num_channels , height, width = input_dim.size()
        #  Changed
        theta = theta.reshape([-1, 3, 3]).float()

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, scale_h)
        grid = grid.unsqueeze(0).reshape([1,-1])
        shape = grid.size()
        grid = grid.expand(num_batch,shape[1])
        grid = grid.reshape([num_batch, 3, -1])

        T_g = torch.matmul(theta, grid)
        x_s = T_g[:,0,:]
        y_s = T_g[:,1,:]
        t_s = T_g[:,2,:]

        t_s_flat = t_s.reshape([-1])

        # smaller
        small = 1e-7
        smallers = 1e-6*(1.0 - torch.ge(torch.abs(t_s_flat), small).float())

        t_s_flat = t_s_flat + smallers
        condition = torch.sum(torch.gt(torch.abs(t_s_flat), small).float())
        # Ty changed
        x_s_flat = x_s.reshape([-1]) / t_s_flat
        y_s_flat = y_s.reshape([-1]) / t_s_flat

        input_transformed = _interpolate( input_dim, x_s_flat, y_s_flat,out_size,scale_h)

        output = input_transformed.reshape([num_batch, out_height, out_width, num_channels ])
        return output, condition

    img_w = U.size()[2]
    img_h = U.size()[1]

    scale_h = True
    output, condition = _transform(theta, U, out_size, scale_h)
    return output, condition


def transform(patch_size_h,patch_size_w,M_tile_inv,H_mat,M_tile,I1,patch_indices,batch_indices_tensor):
    # Transform H_mat since we scale image indices in transformer
    batch_size, num_channels, img_h, img_w = I1.size()
    if torch.cuda.is_available():
        M_tile_inv = M_tile_inv.cuda()
    H_mat = torch.matmul(torch.matmul(M_tile_inv, H_mat), M_tile)
    # Transform image 1 (large image) to image 2
    out_size = (img_h, img_w)
    warped_images, _ = transformer(I1, H_mat, out_size)

    warped_images_flat = warped_images.reshape([-1,num_channels])
    patch_indices_flat = patch_indices.reshape([-1])
    pixel_indices = patch_indices_flat.long() + batch_indices_tensor
    pixel_indices = pixel_indices.unsqueeze(-1).long()
    pixel_indices = pixel_indices.expand(patch_size_h*patch_size_w*batch_size, num_channels)
    pred_I2_flat = torch.gather(warped_images_flat, 0, pixel_indices)

    pred_I2 = pred_I2_flat.reshape([batch_size, patch_size_h, patch_size_w, num_channels])

    return pred_I2.permute(0,3,1,2)

def getBatchHLoss(H, H_inv):
    batch_size = H.size()[0]
    Identity = torch.eye(3)
    if torch.cuda.is_available():
        Identity = Identity.cuda()
    Identity = Identity.unsqueeze(0).expand(batch_size,3,3)
    return criterion_l2(H.bmm(H_inv), Identity)


def display_using_tensorboard(I, I2_ori_img, I2, pred_I2, I2_dataMat_CnnFeature, pred_I2_dataMat_CnnFeature, triMask, loss_map, writer):

    I1_ori_img = cv2.normalize(I.cpu().detach().numpy()[0, 0, ...], None, 0, 255, cv2.NORM_MINMAX,
                               cv2.CV_8U)
    I2_ori_img_ = cv2.normalize(I2_ori_img.cpu().detach().numpy()[0, 0, ...], None, 0, 255, cv2.NORM_MINMAX,
                                cv2.CV_8U)
    input_I2 = cv2.normalize(I2.cpu().detach().numpy()[0, 0, ...], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    pred_I2 = cv2.normalize(pred_I2.cpu().detach().numpy()[0, 0, ...], None, 0, 255, cv2.NORM_MINMAX,
                            cv2.CV_8U)

    I2_channel_1 = cv2.normalize(I2_dataMat_CnnFeature.cpu().detach().numpy()[0, 0, ...], None, 0, 255,
                                 cv2.NORM_MINMAX, cv2.CV_8U)
    pred_I2_channel_1 = cv2.normalize(pred_I2_dataMat_CnnFeature.cpu().detach().numpy()[0, 0, ...], None, 0,
                                      255, cv2.NORM_MINMAX, cv2.CV_8U)

    mask_1 = cv2.normalize(triMask.cpu().detach().numpy()[0, ...], None, 0, 255, cv2.NORM_MINMAX,
                           cv2.CV_8U)
    loss_fig = cv2.normalize(loss_map.cpu().detach().numpy()[0, ...], None, 0, 255, cv2.NORM_MINMAX,
                             cv2.CV_8U)

    writer.add_image('I1 and I2',
                     I1_ori_img,
                     global_step=1,
                     dataformats='HW')
    writer.add_image('I1 and I2',
                     I2_ori_img_,
                     global_step=2,
                     dataformats='HW')

    writer.add_image('I2 and pred_I2',
                     input_I2,
                     global_step=1,
                     dataformats='HW')
    writer.add_image('I2 and pred_I2',
                     pred_I2,
                     global_step=2,
                     dataformats='HW')

    writer.add_image('I2 and pred I2 feature_1',
                     I2_channel_1,
                     global_step=1,
                     dataformats='HW')
    writer.add_image('I2 and pred I2 feature_1',
                     pred_I2_channel_1,
                     global_step=2,
                     dataformats='HW')

    writer.add_image('loss_map and mask',
                     loss_fig,
                     global_step=1,
                     dataformats='HW')
    writer.add_image('loss_map and mask',
                     mask_1,
                     global_step=2,
                     dataformats='HW')


