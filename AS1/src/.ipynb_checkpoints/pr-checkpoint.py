{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import sys\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1"
     ]
    }
   ],
   "source": [
    "if len(sys.argv) == 2:\n",
    "    sys.stdout.write(\"2\")\n",
    "    filename = sys.argv[1]\n",
    "    image = cv.imread(filename)\n",
    "    plt.imshow(image)\n",
    "    cv2.waitKey(0)\n",
    "    cv2.destroyAllWindows()\n",
    "elif len(sys.argv) < 2:\n",
    "    sys.stdout.write(\"1\")\n",
    "    cap = cv2.VideoCapture(0)\n",
    "    retval, image = cap.read()\n",
    "    cv2.imshow(\"image\", image)\n",
    "    cv2.waitKey(0)\n",
    "    cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[NbConvertApp] Converting notebook Untitled.ipynb to script\n",
      "[NbConvertApp] Writing 321 bytes to Untitled.py\n"
     ]
    }
   ],
   "source": [
    "!jupyter nbconvert --to script Untitled.ipynb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
