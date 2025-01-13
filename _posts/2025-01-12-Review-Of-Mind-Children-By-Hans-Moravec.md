---
layout: post
title: Review of Mind Children By Hans Moravec (1988)
---

I've had this book on my reading list for a while since this is the classic book everyone cites about predicting the singularity ahead of time and describing a 'merging' of AI and human minds into the future as a positive singularity. Since I had some time this afternoon, I downloaded and read it (hint: you can borrow it for free from internet archive [here](https://archive.org/details/mindchildren00hans/page/n11/mode/2up)). The book is actually pretty short (only about 160 real pages plus appendices plus bibliography) so can be read in one sitting and is a fairly easy read.

The actual premise of the book is surprisingly straightforward and I actually gained relatively little new knowledge from it. The primary steps are essentially to argue that AI technology is rapidly improving (and indeed at the time (late 1980s) it was and we were in the first AI summer), that Moore's law predicts that we will reach approximate parity with human brain computation in the 2020s (as indeed happened!) and that this will usher in AGI (still tbd but now seems extremely plausible), and that in the end we will reach mind uploading technology and then our new silicon-based minds will continue to evolve and adapt and will drift away from their initial human forms until basically nothing distinguishes us from our AI descendants. Personally, I think this is pretty plausible and, to me, this largely represents the 'good ending' of the singularity, although YMMV. 

In terms of 'AI theology', Moravec seems to have some vague thoughts about RSI but generally seems to envision a slow takeoff with humans and AI systems coexisting and our combined civilization growing and evolving into the stars. He forsees primarily darwinian processes operating on this mass of AIs and human uploads and that this will slowly shape the mind distribution towards 'optimal minds' causing uploaded humans to essentially forget their original human identity and function similarly to the pure AI systems. Moravec generally does not make detailed future predictions of e.g. gdp growth, or effect on current humans and broadly seems to assume that by and by large current systems continue at least long enough for humans to be uploaded into a silicon substrate and hence join the AIs on the capability frontier. He does not discuss AI alignment nor consider it seemingly a problem, at least in the book.

Interestingly, the Moore's law projections are pretty accurate. He predicts specifically about 10 tera-ops and 100 terabits of compute and memory available for approximately <span>$</span>1000 in 2030, and this is also his estimate for the required compute and memory to match that of the human brain and hence reach AGI. He expects this to be possible on supercomputers in 2010. This is shockingly good estimate honestly given it was made nearly 40 years ago. For instance, a 4090 GPU (released in 2022) today has approximately 80 teraflops of fp16 performance and costs about <span>$</span>2000-<span>$</span>4000 today. Arguably this is already under his <span>$</span>1000 limit if we price things in 1988 dollars. For 'memory', the success of his prediction highly depends on whether he means disk or RAM. If disk then certainly 100TB is very achievable today. If RAM then we might have to wait for 2030. I also don't think his supercomputing predictions were that outlandish. It feels very plausible that once we get AGI, it turns out we would have been able to run it (inference, not necessarily training) on a 2010s supercomputer. 

Despite all this, I feel like the book was hyped up to be a bit more than it was. I think this was for me a problem of reading backwards. It is very possible that much of this was extremely novel for the time but has since been absorbed into general knowledge of the LW/transhumanist subcultures and hence is no longer novel to me. Many of these points, i.e. mind uploading, simulation theory, cellular automata (and organisms within the cellular automata 'breaking out' by analyzing correlation patterns to understand our world), are not at all new in today's intellectual environment but could have been very different then. Definitely it is possible to trace a lot of inspiration of e.g. [early Yudkowsky's work](https://www.lesswrong.com/posts/5wMcKNAwB6X4mp9og/that-alien-message)  to similar ideas presented here, and that part of intellectual history was interesting to see from a different angle. 

Speaking of transhumanism, a very weird and unexpected (to me) feature of today is how the discourse has essentially entirely forgotten the original transhumanist/extropian movements of the 90s and 2000s [^1]. This was the milieau that shaped and profoundly influenced the early lesswrong days and now it is almost entirely forgotten -- despite the original core ideas and arguments about the nature of future technology, AI, mind uploading, biological engineering, etc being way more realistic and short-term now than then when they were essentially all sci-fi speculations. I think the general internet shift away from mailing lists and blogs and towards social media played a big part, as well as the general intellectual pessimism regarding technology and internecine political fights of the 2010s culture caused this. Still, it is super weird that an entire distinct subculture can basically entire disappear exactly as the issues it discusses are becoming more relevant.

One other thing that surprised me is that the book contains almost no references to neural networks as the mechanism for AGI, despite a.) neural networks actually being all the rage in the 1980s when backprop was invented and first applied and pioneers like Hinton were doing their seminal work, and b.) Moravec models the time to AGI by reference to the computing power of the human brain, which presupposes a fair bit of probability mass on some kind of brain-like architecture since if you think e.g. AGI can be solved by expert systems or SAT solvers or whatever, why is the compute power of the brain relevant (except perhaps as a super crude upper bound)? 

Even more generally Moravec, surprisingly, does not discuss concrete paths to AGI except as extrapolations of the robotics and symbolic logic systems of this era. Even here he thinks only in problems and not in methods. Now, this might actually be the correct approach for long term prediction, since while methods can change unpredictably, the problems are quite easy to define and it is possible to come up with rates of progress that, perhaps, hold over long time horizons even as individual methods succeed one another. This is perhaps an interesting lesson for long-term forecasting in general. Perhaps here inside-view knowledge is actually harmful since it gives an easy view of the problems but not of the solutions, and hence the outside view 'line on graph keeps going' is just a general better forecasting technique at this low level of granularity. Of course, this is ignoring all the lines on graphs that have changed unexpectedly, see [wtf-happened-in-1971](https://wtfhappenedin1971.com) for a whole bunch of these (albeit very opionated and many arguable). So, our forecasting algorithm comes down to a.) find long-running trends on graphs, b.) assume they continue and work out consequences, c.) hope they continue and if not quietly forget. This sounds cynical, and it is, but at the same time it's unclear if it is feasible to do any better, since having detailed knowledge of specific fields also seems a very bad guide to predicting their future. At least lines on graphs empirically have a nontrivial chance of continuing for a long time, allowing you to sometimes make highly counterintuitive correct predictions.

There are good reviews of fairly obscure 1970s and 1980s AI groups and systems which are interesting as well as a good, although slightly outdated description of the functioning of the retina. A lot of this history is very interesting if you are an AI history nerd like me but otherwise not essential. There are some digressions into basic complexity theory and cellular automata and the like, although none of this discussion would be particularly novel to anybody conversant in the general culture of the early lesswrong years.

One very striking idea, which I had not seen before, was his 'robot bush', of a robot with the shape of a branching tree where each branch has independent actuators and sensors. Each branch branches many times until nearly the atomic level such that this robot has actuators that span many orders of magnitude of scales so it can accomplish both coarse touches but also incredibly fine-grained (down to nanometer level) manipulations. The sensory and cognitive requirements of this level of dexterity would be immense but at the same time it would outclass humans unimaginaby in the manipulation of the physical world. 

He also includes some interesting vignettes on 'near future' technologies and life which are always entertaining to read in retrospect. Some things he correctly predicts (but does not give timeframes for) are:

- Mobile phones (portable microcomputers including phone and camera)
- Mapping software for use in driving akin to Google/Apple maps
- Night-vision goggles (although they don't have wide use outside of niche applications today)
- AR glasses
- Virtual assistants like Siri/Alexa

Another thing is that he envisions extremely flexible and manipulable UIs for both e.g. solving physics problems, designing houses, and general programming which operate on highly abstract units. Some of these exist (such as autoCAD software for architects), and definitely some physics simulators exist but are not widely used in the way he describes. A general trend is that he envisions much more dramatic advancements in UI and UX for computer control and interaction than actually have happened. He also generally envisions significantly more success on the hardware side -- i.e. ubiquitous robotic automation including humanoids by early 20th century, than happened. 

It's super easy to see why he would think this. From his perspective, in the last three decades we have gone from punch cards, to teletype terminals, to GUIs and menus etc with roughly a complete revolution per decade. Why should not this revolution continue? Why should there not have been another 3/4 paradigm shifts as great as the inventuion of the GUI so that now we all work in perfectly crafted VR simulation environments with the most incredibly easy, almost telepathic UI possible? And yet, for the most part computer UX innovations have mostly frozen at where they were in the 80s, at least in their fundamentals.  We have mice and keyboards. We type text into files. We use files and folders and drag and drop them. We have menus and drop-downs.  We have slightly shinier but broadly similar GUIs. We have terminals and both code and run programs by text. We still have almost no voice or tactile inputs to our computers. 
There has been one paradigm shift since then -- touchscreens which are now ubiquitous on mobile but rare for desktop/laptop -- but beyond that almost everything we do on computers would likely have been extremely familiar to a computer user of the late 80s. Today, nearly forty years later I am typing this blog post in much the same way that he describes writing the book on his Macintosh [^2]. This is one very interesting case of a line on a graph that suddenly stopped for no apparent reason.

Some of this is that the 'obvious' next steps in UI paradigms such as VR/AR, and tactile feedback have proven much more challenging to get right and at a widely affordable price than I think anybody envisioned. Today, clearly, there is a good sized VR industry with many headsets on the market but they are not ubiquitously used for day-to-day purposes like mobile phones are. The reason for this seems entirely (from my laypersons understanding) to be due to hardware challenges such as making the headsets not be too bulky, get too hot, have enough resolution etc to be widely useable. Certainly there are some areas of hardware, primarily silicon and transistor physics, but also e.g. solar panels, where there are Moore's law like trends, and other areas where there just are not. 

I still eagerly await widely useable AR glasses as I have been ever since Google Glass came out in 2014 (!), and they still seem the same five years away they always have been. It would be very funny and weird if we reached AGI before we have good VR, but it seems likely that is how things will end up. 

Another reason is that UI innovation, even when technically possible, is either not done or nobody really feels a great need for it. For instance, [Bret Victor](https://worrydream.com) has many extremely cool-looking UI projects which showcase how we can integrate simulation and dynamics and tactile sense into UIs, but as far as I can tell basically none of it has ever taken off and I'm unsure of why. It might just be that UIs are very sticky and that once something reaches 'good enough' then it is very hard to convince people to upgrade. 

In any case, that's enough musings for now. Overall, I think it is a great book to have read, although it is surprising the degree to which the ideas in it have seeped into the general culture and hence have lost their novelty from the vantage point of today. It is always surprising how predictable some things are in advance if you find the right lines on graphs and it really is amazing, when you think about it, that Moore's law has continued unabated for nearly 70 years. Exponential growth, it turns out, can last a surprisingly long time indeed, at least in certain areas.

[^1]: Of course last year there was E/acc which now seems entirely dead and was in any case a shallow and obnoxious twitter fad.

[^2]: In the last few years it has become possible to use LLMs for drafting or editing my blogs, which does constitute a paradigm shift but I have extensively tried using GPT4 and Claude to help with writing and haven't found immense utility from this yet. Likely this will change shortly.