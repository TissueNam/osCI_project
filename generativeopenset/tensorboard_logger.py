# # Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
# import tensorflow as tf
# import numpy as np
# import scipy.misc
# from io import BytesIO
# from PIL import Image
# import torchvision
# import torch 


# class Logger(object):
#     def __init__(self, log_dir):
#         """Create a summary writer logging to log_dir."""
#         self.writer = tf.summary.create_file_writer(log_dir)

#     def scalar_summary(self, tag, value, step):
#         value = value.cpu().clone().numpy()
#         """Add scalar summary."""
#         with self.writer.as_default():
#             tf.summary.scalar(tag, value, step)
#             self.writer.flush()

#     # def scalar_summary(self, tag, value, step):
#     #     """Log a scalar variable."""
#     #     summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#     #     self.writer.add_summary(summary, step)

#     def image_summary(self, tag, image, step,dataformats='HWC'):
#         """Log image , Input image will be a numpy ndarray 
#         dataformats : can be changed as per requirement to HWC,CHW..
#         where H-Height W-Width and C-Channels of image"""
#         self.writer.add_image(tag,image,step,dataformats)
#         self.writer.flush()	

#     # def image_summary(self, tag, images, step):
#     #     """Log a list of images."""

#     #     img_summaries = []
#     #     for i, img in enumerate(images):
#     #         # Write the image to a string
#     #         try:
#     #             s = StringIO()
#     #         except:
#     #             s = BytesIO()
#     #         scipy.misc.toimage(img).save(s, format="png")

#     #         # Create an Image object
#     #         img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
#     #                                    height=img.shape[0],
#     #                                    width=img.shape[1])
#     #         # Create a Summary value
#     #         img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

#     #     # Create and write Summary
#     #     summary = tf.Summary(value=img_summaries)
#     #     self.writer.add_summary(summary, step)

#     def histo_summary(self, tag, values, step, bins=1000):
#         """Log a histogram of the tensor of values."""

#         # Create a histogram using numpy
#         counts, bin_edges = np.histogram(values, bins=bins)

#         # Fill the fields of the histogram proto
#         hist = tf.HistogramProto()
#         hist.min = float(np.min(values))
#         hist.max = float(np.max(values))
#         hist.num = int(np.prod(values.shape))
#         hist.sum = float(np.sum(values))
#         hist.sum_squares = float(np.sum(values ** 2))

#         # Drop the start of the first bin
#         bin_edges = bin_edges[1:]

#         # Add bin edges and counts
#         for edge in bin_edges:
#             hist.bucket_limit.append(edge)
#         for c in counts:
#             hist.bucket.append(c)

#         # Create and write Summary
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
#         self.writer.add_summary(summary, step)
#         self.writer.flush()

from io import BytesIO         # Python 3.x
import numpy as np
from PIL import Image 
import torch 
import torch.utils.tensorboard as tb
import torchvision

class Logger(object):
    def __init__(self, log_dir):
        """ Create a summary writer object logging to log_dir."""
        self.writer = tb.SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag,value, step)
        self.writer.flush()

    def image_summary(self, tag, image, step,dataformats='CHW'):
        """Log image , Input image will be a numpy ndarray 
        dataformats : can be changed as per requirement to HWC,CHW..
        where H-Height W-Width and C-Channels of image"""
        if type(image==list):
            image = np.array(image)
        image_torch = torch.from_numpy(image)
        image_grid = torchvision.utils.make_grid(image_torch[:32], normalize=True)
        self.writer.add_image(tag, image_grid, step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag,values, step)
        self.writer.flush()

    def text_summary(self,tag,value,step):
        """Log text with tag to it"""
        self.writer.add_text(tag,value,step)
        self.writer.flush()

    def embedding_summary(self,embedding_matrix, metadata=None, label_img=None, 
          global_step=None, tag='default', metadata_header=None):
        """Log embedding matrix to tensorboard."""
        self.writer.add_embedding(embedding_matrix, metadata, label_img,global_step, tag,
                metadata_header)
        self.writer.flush()

    def plot_pr_summary(self,tag, labels, predictions, global_step=None,
            num_thresholds=127, weights=None, walltime=None):
        """Plot Precision/Recall curves with labels being actual labels 
        and predictions being how accurarte(in tems of %)"""
        self.writer.add_pr_curve(tag, labels, predictions, global_step, num_thresholds, weights, walltime)
        self.writer.flush()

    def __del__(self):
        """close the writer"""
        self.writer.close()