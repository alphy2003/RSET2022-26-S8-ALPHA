import { createContext, useContext, useEffect, useState } from "react";
import { initializeApp } from "firebase/app";
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, onAuthStateChanged } from "firebase/auth";
import { getFirestore, doc, setDoc, getDoc } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyASECyssLCgDp96CyU2wIl3OEz7LdibX-Q",
  authDomain: "corealign-31a0c.firebaseapp.com",
  projectId: "corealign-31a0c",
  storageBucket: "corealign-31a0c.firebasestorage.app",
  messagingSenderId: "161580398500",
  appId: "1:161580398500:web:97dd814ad8a784cbd03331",
  measurementId: "G-MT6YBM2041"
};

const firebaseapp = initializeApp(firebaseConfig);
const firebaseauth = getAuth(firebaseapp);
const firestore = getFirestore(firebaseapp);

export const FirebaseContext = createContext(null);

export const useFirebase = () => useContext(FirebaseContext);
  
export const FirebaseProvider = (props) => {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(firebaseauth, (user) => {
            setCurrentUser(user);
            setLoading(false);
        });

        return unsubscribe;
    }, []);

    const signup = async (email, password, name) => {
        try {
            const userCredential = await createUserWithEmailAndPassword(firebaseauth, email, password);
            const user = userCredential.user;
            
            // Create user document in Firestore
            await setDoc(doc(firestore, "users", user.uid), {
                id: user.uid,
                name: name,
                email: email,
                dayStreak: 0,
                totalWorkouts: 0,
                avgDurationMinutes: 0,
                weeklyWorkoutsTarget: 3
            });

            return userCredential;
        } catch (error) {
            throw error;
        }
    };

    const signin = (email, password) => {
        return signInWithEmailAndPassword(firebaseauth, email, password);
    };

    const signout = () => {
        return signOut(firebaseauth);
    };

    const getUserData = async (userId) => {
        try {
            const userDocRef = doc(firestore, "users", userId);
            const userDoc = await getDoc(userDocRef);
            
            if (userDoc.exists()) {
                return userDoc.data();
            } else {
                throw new Error("User document not found");
            }
        } catch (error) {
            console.error("Error fetching user data:", error);
            throw error;
        }
    };

    return(
        <FirebaseContext.Provider value={{ signup, signin, signout, currentUser, loading, getUserData }}>
            {props.children}
        </FirebaseContext.Provider>
    );
}