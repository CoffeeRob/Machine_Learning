Overview -
This project tests to see if unsupervised learning can be implemented to create
representations of input data that can be used for classification. The code is
comprised of an autoencoder followed by a classifier. The autoencoder has two parts,
the first being an encoder that takes an input (a batch of handwritten numbers )
and compresses the information down to a latent sized vector. This is followed by
a decoder that tries to reverse the process and reconstruct the image from the
minimal information given by the vector. The objective is to reconstruct the initial
input through this bottleneck. The classifier then takes the trained encoder and
puts a supervised linear after it to take the compressed information and relate
it to the ten possible digits. This linear needs to be trained, so the weights of the encoder are frozen.

Experiments-
Two control experiments were additionally run to provide an upper and lower bound
for how well the unsupervised trainer was expected to do. The lower bound was provided
by running an uninitialised encoder followed by the supervised linear, to check
that the accuracy wasn't just a result of this supervised final layer. The upper bound
was provided by running a supervised train on the encoder and the linear, which
we'd expect to do better than the unsupervised classifier.
Each experiment ran until the change of loss between epochs was reduced to less
than 2.5%, to save time.

Results-
The unsupervised control was about 65-70% accurate after 8 epochs, which might sound
high but is showing that even an uninitialised conv net is helpful as it is translation
invariant.
The supervised classifier was about 98% accurate after 5 epochs.
The unsupervised classifier was about 90% accurate after 3 epochs, which as this
is towards the upper end of the expected range, counts as a roaring success!

Lessons learnt -
By creating a deep net, you are making it more difficult for the net to make sense
of the gradients it receives as the gradients are mixed with each other.

The SGD optimiser just was not good enough to create this unsupervised net so
I switched to ADAM instead.

By setting the latent size layer to a low number, you're forcing the net to compress a lot,
so it has to train well, making the final linear better at taking this to the corresponding digit.
