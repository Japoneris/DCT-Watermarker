import cv2
import numpy as np


class BlockMark:
    def __init__(self, block_size=8, mode="add", alpha=2.):
        """
        :param block_size: size of an embedding block. 8 is recommanded. A power of 2 is recommanded. 
        :param mode: add | multiply. Multiply better adapt to contrast, but more errors if low colors
        :param alpha: mixing coefficient. At least 1, 
        :type alpha: float
        :type block_size: int
        :type mode: str
        """

        # Within the block, use only "mid frequency" terms:
        # - High Frequency are too visible
        # - Low Frequency can be removed by compression
        x = np.arange(block_size)
        R, _ = np.meshgrid(x, x)
        D = R + R.T
        
        # Mid freq are before the diagonal
        self.locs = np.where((D >= block_size //2) & (D < block_size))
        self.l = len(self.locs[0]) # Number of coefficients
        self.size = block_size

        # Which operation to do during embedding
        self.mode = mode
        if mode == "add":
            self.fx = self.add
            self.fx_inv = self.add_inv
        
        elif mode == "multiply":
            self.fx = self.multiply
            self.fx_inv = self.multiply_inv
            
        else:
            assert(False)
            
        self.alpha = alpha
        return

    def add(self, vec, msg_chunk):
        """Add msg_chunk to image vector"""
        return vec + self.alpha * msg_chunk

    def add_inv(self, F0, F1):
        """Recover message by diff between raw and watermarked image vector"""
        return (F1 - F0)/self.alpha
        
        
    def multiply(self, vec, msg_chunk):
        """Embedd message thanks to multiplication"""
        return vec * (1 + self.alpha * msg_chunk)    

    def multiply_inv(self, F0, F1, tol=10e-6):
        """
        :param F0: coefficients of the Non-Watermarked img
        :param F1: coefficients of the Watermarked img
        :param tol: to avoid division by 0
        """
        return (F1 / F0 - 1)/ self.alpha
    
    def get_max_capacity(self, img):
        """Compute the maximal number of bits that can be written in an image
        """
        nr, nc = np.array(img.shape) // self.size
        return self.l * nr * nc

    def get_block_count(self, msg):
        """Compute number of blocks needed to encode the message
        """
        return (len(msg) - 1) // self.l + 1
        
    def encode(self, img, msg):
        """Encode the full message
        
        :param img: grayscale image to watermark
        :param msg: bit vector (0, 1)
        """
        # 0/1 -> [-1, 1] the vector to ease the embedding
        msg = (np.array(msg) - 0.5) * 2
        
        # Pad the vector to get the correct length
        l = self.l
        if len(msg) % l != 0:
            left = l - (len(msg) % l)
            msg = np.concatenate([msg, np.zeros(left)])

        # The image should be enough to encode the message
        n_chunk = len(msg) // l
        nr, nc = np.array(img.shape) // self.size
        assert(n_chunk <= nr * nc)
        
        # Loop over blocks to encode
        img_new = img.astype(np.float32)
        for i in range(n_chunk):
            x_row, x_col = i // nc, i % nc
            self._encode(img_new, msg[l*i:l*(i+1)], x_row, x_col)

        # Re-encode the image properly
        return img_new.round().clip(0,255).astype(np.uint8)

    def _encode(self, img, msg, i, j):
        """Encode over a single block
        
        :param i, j: block location
        :param img: raw img
        :param msg: message chunk to encode
        """
        s = self.size

        # Get the block
        P = img[i*s:(i+1)*s, j*s:(j+1)*s]
        # Pixel => Frequency
        F = cv2.dct(P)

        # Get mid-frequency coefficients
        vec = F[self.locs[0], self.locs[1]]
        
        # Update coefficients
        F[self.locs[0], self.locs[1]] = self.fx(vec, msg)

        # Get pixels back 
        img[i*s:(i+1)*s, j*s:(j+1)*s] = cv2.idct(F)
        return
        
    def decode(self, img_raw, img_wat, quantization=True):
        """Decode the whole image 
        
        :param k: number of blocks
        :param img_raw: non watermarked image
        :param img: watermarked image
        :param quantization: perform full recovery information. Else, return raw info.
        """
        
        nr, nc = np.array(img_raw.shape) // self.size
        # The image should be enough to encode the message
        lst = []
        for i in range(nr * nc):
            x_row, x_col = i // nc, i % nc
            msg = self._decode(img_raw, img_wat, x_row, x_col)
            
            if np.abs(msg).sum() == 0:
                # No -1 / +1 over the block
                break
            
            else:
                lst.append(msg)

        # Search where we have at least k consecutive 0 to stop the search
        if quantization:
            msg_quant =  np.concatenate(lst).round().astype(int)
            l = len(msg_quant)
            while msg_quant[l-1] == 0:
                l -= 1
            return (msg_quant[:l]+1) // 2

        else:
            return np.concatenate(lst)
        

    def _decode(self, img_raw, img_wat, i, j):
        s = self.size
        # Get the blocks.
        P0 = img_raw[i*s:(i+1)*s, j*s:(j+1)*s].astype(np.float32)
        P1 = img_wat[i*s:(i+1)*s, j*s:(j+1)*s].astype(np.float32)

        F0 = cv2.dct(P0)[self.locs[0], self.locs[1]]
        F1 = cv2.dct(P1)[self.locs[0], self.locs[1]]
        DF = self.fx_inv(F0, F1)
        return DF

    