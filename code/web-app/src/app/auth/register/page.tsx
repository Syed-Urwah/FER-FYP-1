'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Spinner } from '@/components/ui/spinner';

export default function Register() {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState('');
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        try {
            setIsLoading(true)
            const res = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name, email, password }),
            });

            setIsLoading(false)
            if (res.ok) {
                router.push('/auth/signin');
            } else {
                const data = await res.json();
                setError(data.message || 'Registration failed');
            }
        } catch (err) {
            setError('An error occurred');
        }
    };

    return (
        <div className="flex min-h-screen items-center justify-center bg-gray-900 text-white p-4">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 opacity-75"></div>
            <div className="relative z-10 flex flex-col items-center">
                <h1 className="text-4xl font-bold mb-8 tracking-tight">Emotion Detector</h1>
                <Card className="w-full max-w-md bg-gray-800 border border-gray-700 shadow-xl">
                <CardHeader className="border-b border-gray-700 pb-4">
                    <CardTitle className="text-3xl font-extrabold text-center text-indigo-300">Create Account</CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                    <form onSubmit={handleSubmit} className="space-y-4">
                        {error && <div className="text-red-400 text-sm text-center mb-4">{error}</div>}
                        <div>
                            <label className="block text-sm font-semibold text-gray-300 mb-1">Name</label>
                            <Input
                                type="text"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                placeholder="John Doe"
                                required
                                className="bg-gray-700 border-gray-600 text-white placeholder-gray-400 focus:ring-indigo-500 focus:border-indigo-500"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-semibold text-gray-300 mb-1">Email</label>
                            <Input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="john@example.com"
                                required
                                className="bg-gray-700 border-gray-600 text-white placeholder-gray-400 focus:ring-indigo-500 focus:border-indigo-500"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-semibold text-gray-300 mb-1">Password</label>
                            <Input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="••••••••"
                                required
                                className="bg-gray-700 border-gray-600 text-white placeholder-gray-400 focus:ring-indigo-500 focus:border-indigo-500"
                            />
                        </div>
                        <Button disabled={isLoading} type="submit" className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition duration-150 ease-in-out">
                            {isLoading ? <Spinner/> : "Register"}
                            
                        </Button>
                        <div className="text-center text-sm text-gray-400 mt-4">
                            Already have an account? <Link href="/auth/signin" className="text-indigo-400 hover:text-indigo-300 font-medium">Sign in</Link>
                        </div>
                    </form>
                </CardContent>
            </Card>
        </div>
        </div>
    );
}
